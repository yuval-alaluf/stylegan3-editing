from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset


class GTResDataset(Dataset):

	def __init__(self, root_path: Path, gt_dir: Path, transform=None, transform_train=None):
		self.pairs = []
		for f in root_path.glob("*"):
			image_path = root_path / f
			gt_path = gt_dir / f
			if f.suffix in [".jpg", ".png", ".jpeg"]:
				self.pairs.append([image_path, gt_path, None])
		self.transform = transform
		self.transform_train = transform_train

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, index):
		from_path, to_path, _ = self.pairs[index]
		from_im = Image.open(from_path).convert('RGB')
		to_im = Image.open(to_path).convert('RGB')
		if self.transform:
			to_im = self.transform(to_im)
			from_im = self.transform(from_im)
		return from_im, to_im
