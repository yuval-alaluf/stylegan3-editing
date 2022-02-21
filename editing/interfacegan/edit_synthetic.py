import sys
from pathlib import Path
from typing import List

import numpy as np
import pyrallis
import torch
from PIL import Image
from dataclasses import dataclass
from pyrallis import field
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from configs.paths_config import model_paths
from editing.interfacegan.face_editor import FaceEditor
from models.stylegan3.model import SG3Generator, GeneratorType
from models.stylegan3.networks_stylegan3 import Generator
from utils.common import make_transform, tensor2im, generate_mp4


INTERFACEGAN_RANGES = {
    "pose": (-4, 5),
    "age": (-5, 5),
    "smile": (-2, 2),
    "Male": (-2, 4)
}
N_TRANSITIONS = 25


@dataclass
class EditConfig:
    # Path to SG3 generator
    generator_path: Path = Path(model_paths["stylegan3_ffhq"])
    # Whether the generator is aligned or unaligned. Used to determine which set of directions to use
    generator_type: GeneratorType = GeneratorType.ALIGNED
    # Where to save the edits and animations to.
    output_path: Path = Path("./edit_results")
    # List of attributes to edit. For example: `age,smile,pose,Male`.
    # Must be an attribute listed in INTERFACEGAN_RANGES
    attributes_to_edit: List[str] = field(default=["age", "smile", "pose", "Male"], is_mutable=True)
    # Number of images to generate and edit for each direction
    n_images_per_edit: int = 100
    # Truncation psi for sampling
    truncation_psi: float = 0.7
    # Whether to randomly apply user-specified transformations when generating images
    apply_random_transforms: bool = False
    # Whether to generate an interpolation animation of the edit
    generate_animation: bool = False
    # Frames-per-second of the generator animation
    fps: int = 25


@pyrallis.wrap()
def main(opts: EditConfig):
    kwargs = {'fps': opts.fps}
    save_path = opts.output_path / str(opts.generator_type)
    save_path.mkdir(exist_ok=True, parents=True)

    generator = SG3Generator(checkpoint_path=opts.generator_path).decoder

    for direction in opts.attributes_to_edit:
        if direction not in INTERFACEGAN_RANGES:
            raise ValueError(f"Given invalid direction {direction}. Must be one of {list(INTERFACEGAN_RANGES.keys())}!")

        print(f"Performing edits on attribute: {direction}")
        direction_output_path = save_path / direction
        direction_output_path.mkdir(exist_ok=True, parents=True)

        for idx in tqdm(range(opts.n_images_per_edit)):
            image, latent = get_random_image(generator, truncation_psi=opts.truncation_psi)
            edit_images, edit_latents = edit(generator=generator,
                                             latent=latent,
                                             direction=direction,
                                             generator_type=opts.generator_type,
                                             apply_user_transformations=opts.apply_random_transforms)
            # save the edited images
            save_coupled_images(edit_images, output_path=direction_output_path / f"{idx}.jpg")
            if opts.generate_animation:
                edit_latents = torch.stack(edit_latents)
                all_images = prepare_animation(latents=edit_latents, generator=generator)
                # duplicate and reverse images for animation
                all_images = all_images + all_images[::-1]
                gif_path = direction_output_path / f"{idx}_animation"
                generate_mp4(gif_path, all_images, kwargs)


def get_random_image(generator: Generator, truncation_psi: float):
    with torch.no_grad():
        z = torch.from_numpy(np.random.randn(1, 512).astype('float32')).to('cuda')
        if hasattr(generator.synthesis, 'input'):
            m = make_transform(translate=(0, 0), angle=0)
            m = np.linalg.inv(m)
            generator.synthesis.input.transform.copy_(torch.from_numpy(m))
        w = generator.mapping(z, None, truncation_psi=truncation_psi)
        img = generator.synthesis(w, noise_mode='const')
        res_image = tensor2im(img[0])
        return res_image, w


def edit(generator: Generator, latent: torch.tensor, direction: str, generator_type: GeneratorType,
         apply_user_transformations: bool = False):
    editor = FaceEditor(generator, generator_type)
    edit_images, edit_latents = editor.edit(latents=latent,
                                            direction=direction,
                                            factor_range=INTERFACEGAN_RANGES[direction],
                                            apply_user_transformations=apply_user_transformations)
    return edit_images, edit_latents


def prepare_animation(latents: List[torch.tensor], generator: Generator, n_transitions: int = N_TRANSITIONS):
    all_images = []
    print("Generating animation...")
    for i in tqdm(range(1, len(latents))):
        alpha_vals = np.linspace(0, 1, n_transitions).tolist()
        for alpha in alpha_vals:
            result_image = get_result_from_vecs(generator, latents[i - 1], latents[i], alpha)[0]
            output_im = tensor2im(result_image)
            all_images.append(np.array(output_im))
    return all_images


def get_result_from_vecs(generator: Generator, vectors_a: torch.tensor, vectors_b: torch.tensor, alpha: float):
    results = []
    for i in range(len(vectors_a)):
        with torch.no_grad():
            cur_vec = vectors_b[i] * alpha + vectors_a[i] * (1 - alpha)
            res = generator.synthesis(cur_vec.unsqueeze(0).cuda(), noise_mode='const')
            results.append(res[0])
    return results


def save_coupled_images(images: List, output_path: Path):
    if type(images[0]) == list:
        images = [image[0] for image in images]
    res = np.array(images[0])
    for image in images[1:]:
        res = np.concatenate([res, image], axis=1)
    res = Image.fromarray(res).convert("RGB")
    res.save(output_path)


if __name__ == '__main__':
    main()
