from pathlib import Path
from typing import Optional, List

from dataclasses import field, dataclass


@dataclass
class InterFaceGANEdit:
    direction: str
    start: int = -5
    end: int = 5


@dataclass
class StyleCLIPEdit:
    target_text: str
    alpha: float
    beta: float

    @property
    def save_name(self):
        return f'result_video_{"_".join(self.target_text.split())}_{self.alpha}_{self.beta}'


@dataclass
class VideoConfig:
    """ All arguments related to inverting and editing videos """

    """ General input/output args """
    # Path to the video to invert and edit
    video_path: Path
    # Path to the trained encoder to use for inversion
    checkpoint_path: Path
    # Path to the output directory
    output_path: Path
    # Path to pre-saved transforms for video (will extract and save to path if doesn't exist)
    landmarks_transforms_path: Optional[Path] = None
    # Optionally add a path to a generator to switch the generator (e.g., after training PTI)
    generator_path: Optional[Path] = None
    # Path to raw video frames to invert (will extract and save to path if doesn't exist)
    raw_frames_path: Optional[Path] = None
    # Path to aligned video frames to invert (will extract and save to path if doesn't exist)
    aligned_frames_path: Optional[Path] = None
    # Path to the cropped frames to invert (will extract and save to path if doesn't exist)
    cropped_frames_path: Optional[Path] = None

    """ Inference args """
    # Number of ReStyle iterations to run per batch
    n_iters_per_batch: int = 3
    # Maximum number of images to invert in video. If None, inverts all images
    max_images: Optional[int] = None

    """ Field of view expansion args """
    # Expansion amounts for field-of-view expansion given in [left, right, top, bottom]
    expansion_amounts: List[int] = field(default_factory=lambda: [0, 0, 0, 0])

    """ Editing args """
    # Comma-separated list of which edit directions to perform with InterFaceGAN
    interfacegan_directions: List[str] = field(default_factory=lambda: ['age'])
    # Comma-separated list of interfacegan ranges for each edit
    interfacegan_ranges: List[str] = field(default_factory=lambda: ['(-4_5)'])
    # Comma-separated list of which edit directions to perform with StyleCLIP
    styleclip_directions: List[str] = field(default_factory=lambda: ["a happy face",
                                                                     "a face with hi-top fade hair",
                                                                     "a face with an afro",
                                                                     "a face with a double chin",
                                                                     "a face with a red lipstick",
                                                                     "a tanned face"])
    # Comma-separated list of alpha and beta values for each edit. Eg., 0.13_4 -> beta=0.13, alpha=4
    styleclip_alpha_betas: List[str] = field(default_factory=lambda: ["(4_0.13)",
                                                                      "(4_0.13)",
                                                                      "(4_0.13)",
                                                                      "(4_0.13)",
                                                                      "(1.5_0.13)",
                                                                      "(3.5_0.13)"])
    interfacegan_edits = None
    styleclip_edits = None

    def __post_init__(self):
        self.interfacegan_edits = self._parse_interfacegan_edits()
        self.styleclip_edits = self._parse_styleclip_edits()

    def _parse_interfacegan_edits(self):
        factor_ranges = self._parse_factor_ranges()
        if len(self.interfacegan_directions) != len(factor_ranges):
            raise ValueError("Invalid edit directions and factor ranges. Please provide a single factor range for each "
                             f"edit direction. Given: {self.interfacegan_directions} and {self.interfacegan_ranges}")
        interfacegan_edits = []
        for edit_direction, factor_range in zip(self.interfacegan_directions, factor_ranges):
            edit = InterFaceGANEdit(direction=edit_direction, start=factor_range[0], end=factor_range[1])
            interfacegan_edits.append(edit)
        return interfacegan_edits

    def _parse_factor_ranges(self):
        factor_ranges = []
        for factor in self.interfacegan_ranges:
            start, end = factor.strip("()").split("_")
            factor_ranges.append((int(start), int(end)))
        return factor_ranges

    def _parse_styleclip_edits(self):
        alpha_betas = self._parse_styleclip_alpha_betas()
        if len(self.styleclip_directions) != len(alpha_betas):
            raise ValueError("Invalid edit directions and alpha-beta pairs. Please provide a single alpha-beta for each"
                             f" edit direction. Given: {self.styleclip_directions} and {self.styleclip_alpha_betas}")
        styleclip_edits = []
        for edit_direction, alpha_beta in zip(self.styleclip_directions, alpha_betas):
            alpha, beta = alpha_beta
            edit = StyleCLIPEdit(target_text=edit_direction, alpha=alpha, beta=beta)
            styleclip_edits.append(edit)
        return styleclip_edits

    def _parse_styleclip_alpha_betas(self):
        alpha_betas = []
        for alpha_beta in self.styleclip_alpha_betas:
            alpha, beta = alpha_beta.strip("()").split("_")
            alpha_betas.append((float(alpha), float(beta)))
        return alpha_betas
