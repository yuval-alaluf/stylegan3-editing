from typing import Tuple, List

import imageio
import numpy as np
import torch
from PIL import Image


def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


def get_identity_transform():
    translate = (0, 0)
    rotate = 0.
    m = make_transform(translate, rotate)
    m = np.linalg.inv(m)
    return m


def generate_random_transform(translate=0.3, rotate=25):
    rotate = np.random.uniform(low=-1 * rotate, high=rotate)
    translate = (np.random.uniform(low=-1 * translate, high=translate),
                 np.random.uniform(low=-1 * translate, high=translate))
    m = make_transform(translate, rotate)
    user_transforms = np.linalg.inv(m)
    return user_transforms


def tensor2im(var: torch.tensor):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def generate_mp4(out_name, images: List[np.ndarray], kwargs):
    writer = imageio.get_writer(str(out_name) + '.mp4', **kwargs)
    for image in images:
        writer.append_data(np.array(image))
    writer.close()
