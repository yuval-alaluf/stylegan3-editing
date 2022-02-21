from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field

from inversion.options.train_options import TrainOptions


@dataclass
class e4eTrainOptions(TrainOptions):
    """ Training args for e4e-based models. """

    """ General args """
    # Dw loss multiplier factor
    w_discriminator_lambda: float = 0
    # Dw learning rate
    w_discriminator_lr: float = 2e-5
    # Weight of the R1 regularization
    r1: float = 10
    # Interval for applying R1 regularization
    d_reg_every: int = 16
    # Whether to store a latent codes pool for the discriminator training
    use_w_pool: bool = True
    # W pool size when using pool for discrminator training
    w_pool_size: int = 50
    # Truncation psi for sampling real latents for discriminator
    truncation_psi: float = 1

    """ e4e modules args """
    # Norm type for delta loss
    delta_norm: int = 2
    # Delta regularization loss weight
    delta_norm_lambda: float = 2e-4

    """ Progressive training args """
    progressive_steps: Optional[List[int]] = None
    progressive_start: Optional[int] = None
    progressive_step_every: Optional[int] = 2000

    """ Saving and resume training args """
    save_training_data: bool = False
    sub_exp_dir: Optional[str] = None
    resume_training_from_ckpt: Optional[Path] = None
    update_param_list: Optional[str] = None
