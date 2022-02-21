import torch
from torch.utils.data import Dataset


class LatentsDataset(Dataset):

	def __init__(self, latents, opts, transforms=None):
		self.latents = latents
		self.transforms = transforms
		self.opts = opts

	def __len__(self):
		return self.latents.shape[0]

	def __getitem__(self, index):
		if self.transforms is not None:
			return self.latents[index], torch.from_numpy(self.transforms[index][3]).float()
		return self.latents[index]
