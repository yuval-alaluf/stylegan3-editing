import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pyrallis
import torch
import torchvision
from dataclasses import dataclass
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from configs.paths_config import model_paths, styleclip_directions
from editing.styleclip_global_directions.global_direction import StyleCLIPGlobalDirection
from models.stylegan3.networks_stylegan3 import Generator
from models.stylegan3.model import SG3Generator
from utils.common import generate_random_transform, get_identity_transform


@dataclass
class EditConfig:
    # Neutral text
    neutral_text: str = "a face"
    # Which text to edit towards
    target_text: str = "a smiling face"
    # Minimum beta value
    beta_min: float = 0.11
    # Maximum beta value
    beta_max: float = 0.16
    # Number of different beta values to use for editing
    num_betas: int = 5
    # Minimum alpha value
    alpha_min: int = -5
    # Maximum alpha value
    alpha_max: int = 5
    # Number of different alpha values to use for editing
    num_alphas: int = 11
    # Path to file containing delta_i_c
    delta_i_c: Path = Path(styleclip_directions["ffhq"]["delta_i_c"])
    # Path to file containing s statistics
    s_statistics: Path = Path(styleclip_directions["ffhq"]["s_statistics"])
    # Path to file containing text prompts
    text_prompt_templates: Path = Path(styleclip_directions["templates"])
    # Path to the pretrained generator
    stylegan_weights: Path = Path(model_paths["stylegan3_ffhq_pt"])
    # Output size of the SG3 generator
    stylegan_size: int = 1024
    # Truncation strength for sampling latents
    stylegan_truncation: int = 0.7
    # Whether the SG3 generator is the landscapes generator provided in the repo (using the t-Config)
    is_landscapes_generator: bool = False
    # Path to output results
    output_path: Path = Path("./experiments")
    # Path to the `latents.npy` file saved with inference_iterative.py. If None, will perform editing on random images
    latents_path: Optional[Path] = None
    # Path to the landmarks transforms used to get the unaligned images
    landmarks_transforms_path: Optional[Path] = None
    # When editing synthetic images, whether to randomly apply user-specified transformations when generating images
    apply_random_transforms: bool = False
    # When editing synthetic images, truncation psi used for sampling
    truncation_psi: float = 0.7
    # Maximum number of images to edit. If None, edit all images.
    n_images: Optional[int] = None


@pyrallis.wrap()
def main(opts: EditConfig):
    stylegan_model = load_stylegan_model(opts)
    global_direction_calculator = load_direction_calculator(stylegan_model, opts)
    opts.output_path = opts.output_path / "styleclip_edits" / f"{opts.neutral_text}_to_{opts.target_text}"
    opts.output_path.mkdir(exist_ok=True, parents=True)
    if opts.latents_path is not None:
        edit_real_images(stylegan_model, global_direction_calculator, opts)
    else:
        edit_synthetic_images(stylegan_model, global_direction_calculator, opts)


def edit_real_images(stylegan_model: Generator, global_direction_calculator: StyleCLIPGlobalDirection, opts: EditConfig):
    # get pre-saved latents from the experiment directory
    latents = np.load(opts.latents_path, allow_pickle=True).item()
    # load the pre-computed landmarks-based transforms
    landmarks_transforms = None
    if opts.landmarks_transforms_path is not None:
        landmarks_transforms = np.load(opts.landmarks_transforms_path, allow_pickle=True).item()

    # edit all images
    for idx, (image_name, latent) in enumerate(latents.items()):
        if opts.n_images is not None and idx > opts.n_images:
            break
        landmarks_transform = None
        if landmarks_transforms is not None:
            landmarks_transform = landmarks_transforms[image_name][-1]

        edit_image(image_name=image_name,
                   latent=latent[-1],   # use the final latent code outputted by ReStyle encoder
                   landmarks_transform=landmarks_transform,
                   stylegan_model=stylegan_model,
                   global_direction_calculator=global_direction_calculator,
                   opts=opts)


def edit_synthetic_images(stylegan_model: Generator, global_direction_calculator: StyleCLIPGlobalDirection, opts: EditConfig):
    assert opts.n_images is not None, "When editing synthetic images, you need to specify the number of " \
                                      "images to generate using `--n_images`. Given None."
    print("Editing synthetic images...")
    for idx in tqdm(range(opts.n_images)):
        with torch.no_grad():
            z = torch.from_numpy(np.random.randn(1, 512).astype('float32')).to('cuda')
            w = stylegan_model.mapping(z, None, truncation_psi=opts.truncation_psi)
        landmarks_transform = get_identity_transform()
        if opts.apply_random_transforms:
            landmarks_transform = generate_random_transform(translate=0.3, rotate=25)
        edit_image(latent=w[0].cpu().numpy(),
                   landmarks_transform=landmarks_transform,
                   stylegan_model=stylegan_model,
                   global_direction_calculator=global_direction_calculator,
                   image_name=f"{idx}.jpg",
                   opts=opts)


def edit_image(latent: np.ndarray, landmarks_transform: Optional[np.ndarray], stylegan_model: Generator,
               global_direction_calculator: StyleCLIPGlobalDirection, opts: EditConfig,
               image_name: Optional[str] = None, save: bool = True):

    if image_name is not None:
        print(f'Editing {image_name}')

    # initialize latent code
    latent_code = torch.from_numpy(latent).cuda()
    latent_code = latent_code.unsqueeze(0)

    if landmarks_transform is not None:
        stylegan_model.synthesis.input.transform = torch.from_numpy(landmarks_transform).cuda().float()

    with torch.no_grad():
        latent_code_s = stylegan_model.synthesis.W2S(latent_code)

    # generate edits
    latent_code_i = {channel: latent_code_s[channel][0].unsqueeze(0) for channel in latent_code_s}

    alphas = np.linspace(opts.alpha_min, opts.alpha_max, opts.num_alphas)
    betas = np.linspace(opts.beta_min, opts.beta_max, opts.num_betas)

    results, latents_results = [], []
    for beta in betas:
        direction = global_direction_calculator.get_delta_s(opts.neutral_text, opts.target_text, beta)

        edited_latent_code_i = [{c: latent_code_i[c] + alpha * direction[c]
                                 for c in latent_code_i} for alpha in alphas]
        edited_latent_code_i = {c: torch.cat([edited_latent_code_i[i][c]
                                              for i in range(opts.num_alphas)]) for c in edited_latent_code_i[0]}

        for b in range(0, edited_latent_code_i['input'].shape[0]):
            edited_latent_code_i_batch = {c: edited_latent_code_i[c][b:b + 1]
                                          for c in edited_latent_code_i}
            with torch.no_grad():
                edited_image = stylegan_model.synthesis(None, all_s=edited_latent_code_i_batch)
                results.append(edited_image)
                latents_results.append(edited_latent_code_i_batch)

    results = torch.cat(results)
    if save:
        torchvision.utils.save_image(results, opts.output_path / f"{image_name.split('.')[0]}.jpg",
                                     normalize=True, range=(-1, 1), padding=0, nrow=opts.num_alphas)
    return results, latents_results


def load_stylegan_model(opts: EditConfig):
    stylegan_model = SG3Generator(checkpoint_path=opts.stylegan_weights,
                                  res=opts.stylegan_size,
                                  config="landscapes" if opts.is_landscapes_generator else None)
    stylegan_model = stylegan_model.decoder
    return stylegan_model


def load_direction_calculator(stylegan_model: Generator, opts: EditConfig):
    delta_i_c = torch.from_numpy(np.load(opts.delta_i_c)).float().cuda()
    with open(opts.s_statistics, "rb") as channels_statistics:
        _, s_mean, s_std = pickle.load(channels_statistics)
    for channel in s_mean:
        s_mean[channel] = torch.from_numpy(s_mean[channel]).float().cuda()
        s_std[channel] = torch.from_numpy(s_std[channel]).float().cuda()
    with open(opts.text_prompt_templates, "r") as templates:
        text_prompt_templates = templates.readlines()
    with torch.no_grad():
        s_avg = stylegan_model.synthesis.W2S(stylegan_model.mapping.w_avg.unsqueeze(0).repeat(1, 16, 1))
    global_direction_calculator = StyleCLIPGlobalDirection(delta_i_c, s_std, text_prompt_templates, s_avg)
    return global_direction_calculator


if __name__ == "__main__":
    main()
