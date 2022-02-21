import pickle
from enum import Enum
from pathlib import Path
from typing import Optional

import torch

from models.stylegan3.networks_stylegan3 import Generator


class GeneratorType(str, Enum):
    ALIGNED = "aligned"
    UNALIGNED = "unaligned"

    def __str__(self):
        return str(self.value)


class SG3Generator(torch.nn.Module):

    def __init__(self, checkpoint_path: Optional[Path] = None, res: int = 1024, config: str = None):
        super(SG3Generator, self).__init__()
        print(f"Loading StyleGAN3 generator from path: {checkpoint_path}")
        if str(checkpoint_path).endswith("pkl"):
            with open(checkpoint_path, "rb") as f:
                self.decoder = pickle.load(f)['G_ema'].cuda()
                print('Done!')
                return
        elif config == "landscape":
            self.decoder = Generator(
                z_dim=512,
                c_dim=0,
                w_dim=512,
                img_resolution=res,
                img_channels=3,
                channel_base=32768,
                channel_max=512,
                magnitude_ema_beta=0.9988915792636801,
                mapping_kwargs={'num_layers': 2}
            ).cuda()
        else:
            self.decoder = Generator(z_dim=512,
                                     c_dim=0,
                                     w_dim=512,
                                     img_resolution=res,
                                     img_channels=3,
                                     channel_base=65536,
                                     channel_max=1024,
                                     conv_kernel=1,
                                     filter_size=6,
                                     magnitude_ema_beta=0.9988915792636801,
                                     output_scale=0.25,
                                     use_radial_filters=True
                                     ).cuda()
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        print('Done!')

    def _load_checkpoint(self, checkpoint_path):
        try:
            self.decoder.load_state_dict(torch.load(checkpoint_path), strict=True)
        except:
            ckpt = torch.load(checkpoint_path)
            ckpt = {k: v for k, v in ckpt.items() if "synthesis.input.transform" not in k}
            self.decoder.load_state_dict(ckpt, strict=False)
