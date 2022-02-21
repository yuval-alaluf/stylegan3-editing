import sys
import time
from pathlib import Path
from typing import Optional, List

import numpy as np
import pyrallis
import torch
from PIL import Image
from dataclasses import dataclass
from torch.utils.data import DataLoader

sys.path.append(".")
sys.path.append("..")

from configs.paths_config import model_paths
from models.stylegan3.model import SG3Generator
from models.stylegan3.networks_stylegan3 import Generator
from inversion.datasets.pti_dataset import PTIDataset
from inversion.scripts.run_pti_images import PTI
from utils.common import tensor2im
from utils.inference_utils import FULL_IMAGE_TRANSFORMS


@dataclass
class RunConfig:
    # Path to a directory which will contain the resulting generator and images
    output_path: Path
    # Path to the images to use as target for tuning the generator
    images_path: Path
    # Path to `npy` containing the inverted latents to be used for tuning the generator
    latents_path: Path
    # Path to the StyleGAN3 generator
    generator_path: Path = Path(model_paths['stylegan3_ffhq_pt'])
    # Path to `npy` with the transformations used for generating the unaligned images
    landmarks_transforms_path: Optional[Path] = None

    # Device to run on
    device: str = "cuda"
    # Optimizer learning rate
    learning_rate: float = 3e-4
    # Batch size in dataloader
    batch_size: int = 4
    # Number of workers in dataloader
    num_workers: int = 2

    # LPIPS loss lambda
    lpips_lambda: float = 1.
    # L2 loss lambda
    l2_lambda: float = 1.
    # Number of optimization steps
    steps: int = 350
    # LPIPS threshold value for early stopping of optimization process
    lpips_threshold: float = 0.06

    # Model saving interval
    model_save_interval: Optional[int] = None
    # Images saving interval
    save_interval: Optional[int] = None
    # Whether to store the final tuned model or not
    save_final_model: bool = True


@pyrallis.wrap()
def main(opts: RunConfig):
    opts.output_path.mkdir(exist_ok=True, parents=True)
    (opts.output_path / 'images').mkdir(exist_ok=True, parents=True)

    generator = SG3Generator(checkpoint_path=opts.generator_path).decoder

    latents_dict = np.load(opts.latents_path, allow_pickle=True).item()
    latents = [latents_dict[image_name] for image_name in latents_dict.keys()]

    input_paths = [opts.images_path / image_name for image_name in latents_dict.keys()]
    targets = [Image.open(path).convert("RGB") for path in input_paths]

    landmarks_transforms = None
    if opts.landmarks_transforms_path is not None:
        print("Using landmarks transformations!")
        landmarks_transforms = np.load(opts.landmarks_transforms_path, allow_pickle=True).item()
        landmarks_transforms = [landmarks_transforms[path.name][-1] for path in input_paths]

    start = time.time()
    VideoPTI(opts).optimize_model(generator=generator,
                                  codes=latents,
                                  target_images=targets,
                                  landmarks_transforms=landmarks_transforms)
    print(f"Total time: {time.time() - start}")


class VideoPTI(PTI):

    def __init__(self, opts: RunConfig):
        super().__init__(opts)

    def optimize_model(self, generator: Generator, codes, target_images,
                       landmarks_transforms: Optional[List[np.ndarray]] = None, image_name: str = None):

        optimizer = self.get_optimizer(generator)

        dataset = PTIDataset(targets=target_images,
                             latents=codes,
                             landmarks_transforms=landmarks_transforms,
                             transforms=FULL_IMAGE_TRANSFORMS)
        dataloader = DataLoader(dataset,
                                batch_size=self.opts.batch_size,
                                shuffle=True,
                                num_workers=self.opts.num_workers,
                                drop_last=False)

        step = 0
        while step < self.opts.steps:

            for batch_idx, batch in enumerate(dataloader):

                batch_landmarks_transforms = None
                if landmarks_transforms is not None:
                    targets, latents, batch_landmarks_transforms, indices = batch
                else:
                    targets, latents, indices = batch

                targets, latents = targets.to(self.device).float(), latents.to(self.device).float()

                if batch_landmarks_transforms is not None:
                    generator.synthesis.input.transform = batch_landmarks_transforms.cuda().float()

                outputs = generator.synthesis(latents, noise_mode='const', force_fp32=True)

                loss, lpips_loss, l2_loss_val = self.calc_loss(outputs, targets)

                if lpips_loss is not None and lpips_loss < self.opts.lpips_threshold:
                    break

                description = self.get_description(step, loss, lpips_loss, l2_loss_val)
                print(description)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.opts.save_interval is not None and step % self.opts.save_interval == 0:
                    for idx in range(outputs.shape[0]):
                        new_image_path = self.opts.output_path / 'images' / f'step_{step}_{indices[idx]}.jpg'
                        out_image = tensor2im(outputs[idx]).resize((512, 512))
                        target_image = tensor2im(targets[idx]).resize((512, 512))
                        coupled_image = np.concatenate([np.array(target_image), np.array(out_image)], axis=1)
                        Image.fromarray(coupled_image).save(new_image_path)

                if self.opts.model_save_interval is not None and step > 0 and step % self.opts.model_save_interval == 0:
                    model_save_path = self.opts.output_path / f'pti_model_step_{step}.pt'
                    torch.save(generator.state_dict(), model_save_path)

                step += 1

                if step == self.opts.steps:
                    print('OMG, finished training!')
                    break

        # save final images
        for idx in range(outputs.shape[0]):
            final_image_path = self.opts.output_path / 'images' / f'final_{indices[idx]}.jpg'
            out_image = tensor2im(outputs[idx]).resize((512, 512))
            target_image = tensor2im(targets[idx]).resize((512, 512))
            coupled_image = np.concatenate([np.array(target_image), np.array(out_image)], axis=1)
            Image.fromarray(coupled_image).save(final_image_path)

        if self.opts.save_final_model:
            model_save_path = self.opts.output_path / f'final_pti_model.pt'
            torch.save(generator.state_dict(), model_save_path)


if __name__ == '__main__':
    main()
