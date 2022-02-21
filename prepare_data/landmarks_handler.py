from pathlib import Path
from typing import Optional, List

import dlib
import numpy as np
from tqdm import tqdm

from configs.paths_config import model_paths
from utils.alignment_utils import get_stylegan_transform


class LandmarksHandler:
    """
    Computes the landmarks-based transforms between the given aligned and cropped video frames. If the landmarks
    have already been saved to the given `landmarks_transforms_path`, simply load and return them. If they have not
    been saved yet, save them to the `landmarks_transforms_path` for next time.
    """
    def __init__(self, output_path: Path, landmarks_transforms_path: Optional[Path] = None):
        if landmarks_transforms_path is None:
            landmarks_transforms_path = output_path / "landmarks_transforms.npy"
        self.landmarks_transforms_path = landmarks_transforms_path

    def get_landmarks_transforms(self, input_paths: List[Path], cropped_frames_path: Path,
                                 aligned_frames_path: Path, force_computing: bool = False):
        if self.landmarks_transforms_path is None:
            return None
        else:
            if self.landmarks_transforms_path.exists() and not force_computing:
                print(f"Using pre-computed landmarks from path: {self.landmarks_transforms_path}")
                landmarks_transforms = np.load(str(self.landmarks_transforms_path), allow_pickle=True).item()
            else:
                landmarks_transforms = self._compute_landmarks_transforms(input_paths,
                                                                          cropped_frames_path,
                                                                          aligned_frames_path)
                np.save(str(self.landmarks_transforms_path), landmarks_transforms)
            return landmarks_transforms

    @staticmethod
    def _compute_landmarks_transforms(input_paths: List[Path], cropped_frames_path: Path, aligned_frames_path: Path):
        print("Computing landmarks transforms...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(str(model_paths['shape_predictor']))
        landmarks_transforms = {}
        for path in tqdm(input_paths):
            cropped_path = cropped_frames_path / path.name
            aligned_path = aligned_frames_path / path.name
            res = get_stylegan_transform(str(cropped_path), str(aligned_path), detector, predictor)
            if res is None:
                print(f"Failed on: {cropped_path}")
                continue
            else:
                rotation_angle, translation, transform, inverse_transform = res
                landmarks_transforms[path.name] = (rotation_angle, translation, transform, inverse_transform)
        return landmarks_transforms
