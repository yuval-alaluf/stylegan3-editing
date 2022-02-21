import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyrallis
import torch
from PIL import Image
from dataclasses import dataclass
from torch import nn
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from configs.paths_config import model_paths
from models.stylegan3.model import SG3Generator
from models.stylegan3.networks_stylegan3 import Generator
from criteria.lpips.lpips import LPIPS
from utils.common import tensor2im
from utils.inference_utils import FULL_IMAGE_TRANSFORMS


@dataclass
class RunConfig:
    # Path to a directory which will contain the resulting generator and images
    output_path: Path
    # Path to the images to use as target for tuning the generator
    images_path: Path
    # Path to npy containing the inverted latents to be used for tuning the generator
    latents_path: Path
    # Path to the StyleGAN3 generator
    generator_path: Path = Path(model_paths['stylegan3_ffhq_pt'])
    # Path to `npy` with the transformations used for generating the unaligned images
    landmarks_transforms_path: Optional[Path] = None
    # Number of images to run PTI on. If None, runs on all images in images_path
    n_images: Optional[int] = None
    # Device to run on
    device: str = "cuda"
    # Optimizer learning rate
    learning_rate: float = 3e-4
    # LPIPS loss lambda
    lpips_lambda: float = 1.
    # L2 loss lambda
    l2_lambda: float = 1.
    # Number of optimization steps 
    steps: int = 350
    # LPIPS threshold value for early stopping of optimization process
    lpips_threshold: float = 0.06
    # Batch size in dataloader
    batch_size: int = 1
    # Number of workers in dataloader
    num_workers: int = 1
    # How often to save intermediate reconstruction outputs to
    save_interval: Optional[int] = None
    # Whether to save the final tuned model or no
    save_final_model: bool = False
    # Model save interval
    model_save_interval: Optional[int] = None


@pyrallis.wrap()
def main(opts: RunConfig):
    opts.output_path.mkdir(exist_ok=True, parents=True)
    (opts.output_path / 'images').mkdir(exist_ok=True, parents=True)

    # select images based on the pre-saved latents outputted from the encoder
    latents = np.load(opts.latents_path, allow_pickle=True).item()
    input_names = [f for f in opts.images_path.glob("*") if f.name in latents.keys()]
    input_paths = [opts.images_path / f.name for f in input_names]

    landmarks_transforms = None
    if opts.landmarks_transforms_path is not None:
        print("Using landmarks transformations!")
        landmarks_transforms = np.load(opts.landmarks_transforms_path, allow_pickle=True).item()

    print(f'Running PTI on {len(input_paths)} images!')

    for idx, input_path in enumerate(input_paths):
        if opts.n_images is not None and idx >= opts.n_images:
            break
        print(f"Running on {input_path}")
        image_name = input_path.name
        target = Image.open(input_path).convert("RGB")
        # use the last predicted latent of the ReStyle encoder
        latent = latents[image_name][-1]
        # load the images landmarks transforms if training on the unaligned images
        landmarks_transform = landmarks_transforms[image_name][-1] if landmarks_transforms is not None else None

        # reset the generator for every image
        generator = SG3Generator(checkpoint_path=opts.generator_path).decoder

        start = time.time()
        PTI(opts).optimize_model(generator=generator,
                                 codes=latent,
                                 target_images=target,
                                 landmarks_transforms=landmarks_transform,
                                 image_name=image_name)
        print(f"Total time: {time.time() - start}")


class PTI:

    def __init__(self, opts: RunConfig):
        self.opts = opts
        self.device = opts.device
        self.mse_loss = nn.MSELoss().to(self.device).eval()
        self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()

    def get_optimizer(self, generator: Generator):
        # do not alter the fourier features
        params = list(generator.synthesis.parameters())[3:]
        return torch.optim.Adam(params, lr=self.opts.learning_rate)

    def optimize_model(self, generator: Generator, codes: np.ndarray, target_images: Image,
                       landmarks_transforms: Optional[np.ndarray] = None, image_name: Optional[str] = None):

        optimizer = self.get_optimizer(generator)

        image_name = image_name.split(".")[0]
        latents = torch.from_numpy(codes).cuda().unsqueeze(0)
        targets = FULL_IMAGE_TRANSFORMS(target_images).unsqueeze(0).cuda()
        outputs = None

        if landmarks_transforms is not None:
            generator.synthesis.input.transform = torch.from_numpy(landmarks_transforms).cuda().float()

        pbar = tqdm(range(self.opts.steps))
        for step in pbar:

            outputs = generator.synthesis(latents, noise_mode='const', force_fp32=True)

            loss, lpips_loss, l2_loss_val = self.calc_loss(outputs, targets)

            if lpips_loss is not None and lpips_loss < self.opts.lpips_threshold:
                break

            description = self.get_description(step, loss, lpips_loss, l2_loss_val)
            pbar.set_description(description)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.opts.save_interval is not None and step % self.opts.save_interval == 0:
                new_image_path = self.opts.output_path / 'images' / f'step_{step}_{image_name}.jpg'
                out_image = tensor2im(outputs[0]).resize((512, 512))
                target_image = tensor2im(targets[0]).resize((512, 512))
                coupled_image = np.concatenate([np.array(target_image), np.array(out_image)], axis=1)
                Image.fromarray(coupled_image).save(new_image_path)

            if step == self.opts.steps:
                print('OMG, finished training!')
                break

        # save final image
        final_image_path = self.opts.output_path / 'images' / f'final_{image_name}.jpg'
        out_image = tensor2im(outputs[0]).resize((512, 512))
        target_image = tensor2im(targets[0]).resize((512, 512))
        coupled_image = np.concatenate([np.array(target_image), np.array(out_image)], axis=1)
        Image.fromarray(coupled_image).save(final_image_path)

        if self.opts.save_final_model:
            model_save_path = self.opts.output_path / f'final_pti_model_{image_name}.pt'
            torch.save(generator.state_dict(), model_save_path)

    def calc_loss(self, generated_images: torch.tensor, real_images: torch.tensor):
        loss = 0.0
        loss_lpips = None
        l2_loss_val = None
        if self.opts.l2_lambda > 0:
            l2_loss_val = self.mse_loss(generated_images, real_images)
            loss += l2_loss_val * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss += loss_lpips * self.opts.lpips_lambda
        return loss, loss_lpips, l2_loss_val

    @staticmethod
    def get_description(step: int, loss: torch.tensor, lpips_loss: torch.tensor, l2_loss_val: torch.tensor):
        desc = f'Step: {step} - Loss: {loss.item():.4f}'
        if lpips_loss is not None:
            desc += f', LPIPS: {lpips_loss.item():.4f}'
        if l2_loss_val is not None:
            desc += f', L2: {l2_loss_val.item():.4f}'
        return desc


if __name__ == '__main__':
    main()
