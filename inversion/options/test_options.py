from pathlib import Path
from typing import Optional, List

import dataclasses
from dataclasses import dataclass
from pyrallis import field


@dataclass
class TestOptions:
    """ Defines all inference arguments. """

    """ General args """
    # Path to output inference results to
    output_path: Path = Path("./experiments/inference")
    # Path to the pretrained encoder
    checkpoint_path: Path = Path("./experiments/checkpoints/best_model.pt")
    # Path to images to run inference on
    data_path: Path = Path("./gt_images")
    # Whether to resize output images to 256. By default, keeps original resolution
    resize_outputs: bool = False
    # Batch size for running inference
    test_batch_size: int = 2
    # Number of workers for test dataloader
    test_workers: int = 2
    # Number of images to run inference on. If None, runs inference on all images
    n_images: Optional[int] = None
    # Number of forward passes per batch during inference
    n_iters_per_batch: int = 3
    # Path to pkl file with landmarks-based transformations for unaligned images
    landmarks_transforms_path: Optional[Path] = None

    """ Editing args """
    # List of edits to perform
    edit_directions: List[str] = field(default=["age", "smile", "pose"], is_mutable=True)
    # List of ranges for each edit. For example, (-4_5) defines an editing range from -4 to 5
    factor_ranges: List[str] = dataclasses.field(default_factory=lambda: ["(-5_5)", "(-5_5)", "(-5_5)"])

    def __post_init__(self):
        self.factor_ranges = self._parse_factor_ranges()
        if len(self.edit_directions) != len(self.factor_ranges):
            raise ValueError("Invalid edit directions and factor ranges. Please provide a single factor range for each"
                             f"edit direction. Given: {self.edit_directions} and {self.factor_ranges}")

    def _parse_factor_ranges(self):
        factor_ranges = []
        for factor in self.factor_ranges:
            start, end = factor.strip("()").split("_")
            factor_ranges.append((int(start), int(end)))
        return factor_ranges
