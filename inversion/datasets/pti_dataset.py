import numpy as np
import torch
from torch.utils.data import Dataset


class PTIDataset(Dataset):

    def __init__(self, latents, targets, landmarks_transforms=None, transforms=None):
        self.latents = latents
        self.targets = targets
        if landmarks_transforms is not None:
            self.landmarks_transforms = []
            for t in landmarks_transforms:
                if type(t) == np.ndarray:
                    t = torch.from_numpy(t)
                self.landmarks_transforms.append(t.cpu().numpy())
        else:
            self.landmarks_transforms = None
        self.transforms = transforms

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        latent = self.latents[index]
        target = self.targets[index]
        landmarks_transforms = self.landmarks_transforms[index] if self.landmarks_transforms is not None else None
        if self.transforms is not None:
            target = self.transforms(target)
        if self.landmarks_transforms is not None:
            return target, latent, landmarks_transforms, index
        else:
            return target, latent, index
