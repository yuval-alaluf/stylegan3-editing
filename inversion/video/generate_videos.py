from typing import List

import numpy as np

from inversion.video.video_config import VideoConfig
from inversion.video.video_handler import VideoHandler
from utils.common import generate_mp4

OUTPUT_SIZE = (1024, 1024)


def generate_reconstruction_videos(input_images: List, result_images: List, result_images_smoothed: List,
                                   video_handler: VideoHandler, opts: VideoConfig):
    kwargs = {'fps': video_handler.fps}

    # save the original cropped input
    output_path = opts.output_path / 'input_video'
    generate_mp4(output_path, [np.array(im) for im in input_images], kwargs)

    # generate video of original reconstruction (without smoothing)
    output_path = opts.output_path / 'result_video'
    generate_mp4(output_path, result_images, kwargs)

    # generate video of smoothed reconstruction
    output_path_smoothed = opts.output_path / "result_video_smoothed"
    generate_mp4(output_path_smoothed, result_images_smoothed, kwargs)

    # generate coupled video of original frames and smoothed side-by-side
    coupled_images = []
    for im, smooth_im in zip(input_images[2:-2], result_images_smoothed):
        height, width = smooth_im.shape[:2]
        coupled_im = np.concatenate([im.resize((height, height)), smooth_im], axis=1)
        coupled_images.append(coupled_im)
    output_path_coupled = opts.output_path / "result_video_coupled"
    generate_mp4(output_path_coupled, coupled_images, kwargs)
