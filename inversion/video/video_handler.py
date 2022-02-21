from pathlib import Path
from typing import List

import cv2
import dlib
from PIL import Image
from tqdm import tqdm

from prepare_data.preparing_faces_parallel import SHAPE_PREDICTOR_PATH
from utils.alignment_utils import align_face, get_alignment_transformation, get_alignment_positions


class VideoHandler:
    """ Parses a given video and stores the raw, aligned, and cropped video frames. """
    def __init__(self, video_path: Path, output_path: Path, raw_frames_path: Path = None,
                 aligned_frames_path: Path = None, cropped_frames_path: Path = None):
        self.video = cv2.VideoCapture(str(video_path))
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.raw_frames_path = output_path / "raw_frames" if raw_frames_path is None else raw_frames_path
        self.aligned_frames_path = output_path / "aligned_frames" if aligned_frames_path is None else aligned_frames_path
        self.cropped_frames_path = output_path / "cropped_frames" if cropped_frames_path is None else cropped_frames_path
        self.raw_frames_path.mkdir(exist_ok=True, parents=True)
        self.cropped_frames_path.mkdir(exist_ok=True, parents=True)
        self.aligned_frames_path.mkdir(exist_ok=True, parents=True)

    def parse_video(self):
        """ Gets the raw, aligned, and cropped video frames. If they are already saved, uses the pre-saved images. """
        # get raw video frames
        if len(list(self.raw_frames_path.glob("*"))) == 0:
            frames_paths = self._parse_raw_video_frames()
        else:
            frames_paths = [self.raw_frames_path / f for f in self.raw_frames_path.iterdir()]
        # get aligned video frames
        if len(list(self.aligned_frames_path.glob("*"))) == 0:
            self._save_aligned_video_frames(frames_paths)
        else:
            print(f"Aligned video frames already saved to: {self.aligned_frames_path}")
        # get all cropped video frames
        if len(list(self.cropped_frames_path.glob("*"))) == 0:
            self._save_cropped_video_frames(frames_paths)
        else:
            print(f"Cropped video frames already saved to: {self.cropped_frames_path}")

    def get_input_paths(self):
        sorted_paths = sorted(self.aligned_frames_path.iterdir(), key=lambda x: int(str(x.name).replace(".jpg", "")))
        file_names = [f.name for f in sorted_paths]
        aligned_paths = [self.aligned_frames_path / file_name for file_name in file_names]
        cropped_paths = [self.cropped_frames_path / file_name for file_name in file_names]
        return aligned_paths, cropped_paths

    @staticmethod
    def load_images(input_paths: List[Path]):
        input_images = [Image.open(input_path).convert("RGB") for input_path in input_paths]
        return input_images

    def _parse_raw_video_frames(self):
        frames_paths = []
        print("Parsing video!")
        for frame_idx in tqdm(range(self.frame_count)):
            ret, frame = self.video.read()
            if not ret:
                continue
            im_path = self.raw_frames_path / f"{str(frame_idx).zfill(4)}.jpg"
            Image.fromarray(frame[:, :, ::-1]).convert("RGB").save(im_path)
            frames_paths.append(im_path)
        return frames_paths

    def _save_aligned_video_frames(self, frames_paths: List[Path]):
        print("Saving aligned video frames...")
        predictor = dlib.shape_predictor(str(SHAPE_PREDICTOR_PATH))
        detector = dlib.get_frontal_face_detector()
        for path in tqdm(frames_paths):
            try:
                im = align_face(filepath=str(path), detector=detector, predictor=predictor).convert("RGB")
                im.save(self.aligned_frames_path / path.name)
            except Exception as e:
                print(e)
                continue

    def _save_cropped_video_frames(self, frames_paths: List[Path]):
        print("Saving cropped video frames...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(str(SHAPE_PREDICTOR_PATH))
        # crops all the video frames according to the alignment of the first frame
        c, x, y = get_alignment_positions(str(frames_paths[0]), detector, predictor)
        alignment_transform, _ = get_alignment_transformation(c, x, y)
        alignment_transform = (alignment_transform + 0.5).flatten()
        for path in tqdm(frames_paths):
            try:
                curr_im = Image.open(path)
                curr_im = curr_im.transform((1024, 1024), Image.QUAD, alignment_transform, Image.BILINEAR)
                curr_im.save(self.cropped_frames_path / path.name)
            except Exception as e:
                print(e)
                continue
