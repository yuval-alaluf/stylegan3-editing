from pathlib import Path

import pyrallis
from dataclasses import dataclass

from prepare_data.landmarks_handler import LandmarksHandler


@dataclass
class Options:

    """ Input Args """
    # Path to raw images
    raw_root: Path
    # Path to aligned images
    aligned_root: Path
    # Path to cropped images
    cropped_root: Path

    """ Output Args """
    # Path to output directory
    output_root: Path

    """ General Args """
    # Replacing the landmarks if file already exist
    replace: bool = True


@pyrallis.wrap()
def main(args: Options):
    args.output_root.mkdir(exist_ok=True, parents=True)
    landmarks_handler = LandmarksHandler(args.output_root)
    input_images = list(args.raw_root.iterdir())
    landmarks_handler.get_landmarks_transforms(input_paths=input_images,
                                               cropped_frames_path=args.cropped_root,
                                               aligned_frames_path=args.aligned_root,
                                               force_computing=args.replace)


if __name__ == '__main__':
    main()