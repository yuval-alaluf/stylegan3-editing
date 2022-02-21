"""
Code adopted from pix2pixHD (https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py)
"""
from pathlib import Path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename: Path):
    return any(str(filename).endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir: Path):
    images = []
    assert dir.is_dir(), '%s is not a valid directory' % dir
    for fname in dir.glob("*"):
        if is_image_file(fname):
            path = dir / fname
            images.append(path)
    return images
