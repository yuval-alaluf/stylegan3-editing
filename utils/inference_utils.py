from pathlib import Path
from typing import Optional

import dataclasses
import torch
from torchvision import transforms

from configs.paths_config import model_paths
from inversion.models.e4e3 import e4e
from inversion.models.psp3 import pSp
from inversion.options.e4e_train_options import e4eTrainOptions
from inversion.options.test_options import TestOptions
from inversion.options.train_options import TrainOptions
from models.stylegan3.model import SG3Generator
from utils.model_utils import ENCODER_TYPES

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

FULL_IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def load_encoder(checkpoint_path: Path, test_opts: Optional[TestOptions] = None, generator_path: Optional[Path] = None):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts["checkpoint_path"] = checkpoint_path

    if opts['stylegan_weights'] == Path(model_paths["stylegan3_ffhq"]):
        opts['stylegan_weights'] = Path(model_paths["stylegan3_ffhq_pt"])
    if opts['stylegan_weights'] == Path(model_paths["stylegan3_ffhq_unaligned"]):
        opts['stylegan_weights'] = Path(model_paths["stylegan3_ffhq_unaligned_pt"])

    if opts["encoder_type"] in ENCODER_TYPES['pSp']:
        opts = TrainOptions(**opts)
        if test_opts is not None:
            opts.update(dataclasses.asdict(test_opts))
        net = pSp(opts)
    else:
        opts = e4eTrainOptions(**opts)
        if test_opts is not None:
            opts.update(dataclasses.asdict(test_opts))
        net = e4e(opts)

    print('Model successfully loaded!')
    if generator_path is not None:
        print(f"Updating SG3 generator with generator from path: {generator_path}")
        net.decoder = SG3Generator(checkpoint_path=generator_path).decoder

    net.eval()
    net.cuda()
    return net, opts


def get_average_image(net):
    avg_image = net(net.latent_avg.repeat(16, 1).unsqueeze(0).cuda(),
                    input_code=True,
                    return_latents=False)[0]
    avg_image = avg_image.to('cuda').float().detach()
    return avg_image


def run_on_batch(inputs: torch.tensor, net, opts: TrainOptions, avg_image: torch.tensor,
                 landmarks_transform: Optional[torch.tensor] = None):
    results_batch = {idx: [] for idx in range(inputs.shape[0])}
    results_latent = {idx: [] for idx in range(inputs.shape[0])}
    y_hat, latent = None, None
    if "resize_outputs" not in dataclasses.asdict(opts):
        opts.resize_outputs = False

    for iter in range(opts.n_iters_per_batch):
        if iter == 0:
            avg_image_for_batch = avg_image.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
            x_input = torch.cat([inputs, avg_image_for_batch], dim=1)
        else:
            x_input = torch.cat([inputs, y_hat], dim=1)

        is_last_iteration = iter == opts.n_iters_per_batch - 1

        res = net.forward(x_input,
                          latent=latent,
                          landmarks_transform=landmarks_transform,
                          return_aligned_and_unaligned=True,
                          return_latents=True,
                          resize=opts.resize_outputs)

        # if no landmark transforms are given, return the aligned output image
        if landmarks_transform is None:
            y_hat, latent = res

        # otherwise, if current iteration is not the last, return the aligned output; else return final unaligned output
        else:
            # note: res = images, unaligned_images, codes
            if is_last_iteration:
                _, y_hat, latent = res
            else:
                y_hat, _, latent = res

        # store intermediate outputs
        for idx in range(inputs.shape[0]):
            results_batch[idx].append(y_hat[idx])
            results_latent[idx].append(latent[idx].cpu().numpy())

        # resize input to 256 before feeding into next iteration
        y_hat = net.face_pool(y_hat)

    return results_batch, results_latent
