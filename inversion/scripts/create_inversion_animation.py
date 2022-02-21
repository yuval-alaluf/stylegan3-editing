import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
import pyrallis
import torch
from PIL import Image
from dataclasses import dataclass
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from configs.paths_config import model_paths
from models.stylegan3.model import SG3Generator
from models.stylegan3.networks_stylegan3 import Generator
from utils.common import tensor2im, generate_mp4


RESIZE_AMOUNT = (1024, 1024)
N_TRANSITIONS = 25
SIZE = RESIZE_AMOUNT[0]


@dataclass
class RunConfig:
    # Where to save the animations
    output_path: Path
    # Path to directory of images to add to the animation
    data_path: Path
    # Path to `npy` file containing the inverted latents
    latents_path: Path
    # Path to StyleGAN3 generator
    generator_path: Path = Path(model_paths["stylegan3_ffhq"])
    # Path to `npy` with the transformations used for generating the unaligned images
    landmarks_transforms_path: Optional[Path] = None
    # Number of images to include in animation. If None, run on all data
    n_images: Optional[int] = None
    # Fps of the generated animations
    fps: int = 15


@pyrallis.wrap()
def main(opts: RunConfig):
    decoder = SG3Generator(checkpoint_path=opts.generator_path).decoder

    latents = np.load(opts.latents_path, allow_pickle=True).item()
    landmarks_transforms = np.load(opts.landmarks_transforms_path, allow_pickle=True).item()
    image_names = list(latents.keys())
    if opts.n_images is not None:
        image_names = np.random.choice(image_names, size=opts.n_images, replace=False)
    image_paths = [opts.data_path / image_name for image_name in image_names]

    in_images = []
    all_vecs = []
    all_landmarks_transforms = []
    for image_path in image_paths:
        print(f'Working on {image_path.name}...')
        original_image = Image.open(image_path).convert("RGB")
        latent = latents[image_path.name][-1]
        landmark_transform = landmarks_transforms[image_path.name][-1]
        all_vecs.append([latent])
        all_landmarks_transforms.append(landmark_transform)
        in_images.append(original_image.resize(RESIZE_AMOUNT))

    image_paths.append(image_paths[0])
    all_vecs.append(all_vecs[0])
    all_landmarks_transforms.append(all_landmarks_transforms[0])
    in_images.append(in_images[0])

    all_images = []
    for i in range(1, len(image_paths)):
        if i == 0:
            alpha_vals = [0] * 10 + np.linspace(0, 1, N_TRANSITIONS).tolist() + [1] * 5
        else:
            alpha_vals = [0] * 5 + np.linspace(0, 1, N_TRANSITIONS).tolist() + [1] * 5

        for alpha in tqdm(alpha_vals):
            image_a = np.array(in_images[i - 1])
            image_b = np.array(in_images[i])
            image_joint = np.zeros_like(image_a)
            up_to_row = int((SIZE - 1) * alpha)
            if up_to_row > 0:
                image_joint[:(up_to_row + 1), :, :] = image_b[((SIZE - 1) - up_to_row):, :, :]
            if up_to_row < (SIZE - 1):
                image_joint[up_to_row:, :, :] = image_a[:(SIZE - up_to_row), :, :]

            result_image = get_result_from_vecs(decoder,
                                                all_vecs[i - 1], all_vecs[i],
                                                all_landmarks_transforms[i - 1], all_landmarks_transforms[i],
                                                alpha)[0]

            output_im = tensor2im(result_image)
            res = np.concatenate([image_joint, np.array(output_im)], axis=1)
            all_images.append(res)

        kwargs = {'fps': opts.fps}
        opts.output_path.mkdir(exist_ok=True, parents=True)
        gif_path = opts.output_path / f"inversions_gif"
        generate_mp4(gif_path, all_images, kwargs)


def get_result_from_vecs(generator: Generator, vectors_a: List[np.ndarray], vectors_b: List[np.ndarray],
                         landmarks_a: np.ndarray, landmarks_b: np.ndarray, alpha: float):
    results = []
    for i in range(len(vectors_a)):
        with torch.no_grad():
            cur_vec = vectors_b[i] * alpha + vectors_a[i] * (1 - alpha)
            landmarks_transform = landmarks_b * alpha + landmarks_a * (1 - alpha)
            generator.synthesis.input.transform = torch.from_numpy(landmarks_transform).float().cuda().unsqueeze(0)
            res = generator.synthesis(torch.from_numpy(cur_vec).cuda().unsqueeze(0), noise_mode='const', force_fp32=True)
            results.append(res[0])
    return results


if __name__ == '__main__':
    main()
