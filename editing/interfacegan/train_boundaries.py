import sys
from pathlib import Path

import numpy as np
import pyrallis
from dataclasses import dataclass
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from editing.interfacegan.helpers.anycostgan import attr_list
from editing.interfacegan.helpers.manipulator import train_boundary


@dataclass
class TrainConfig: 
    # Path to the `npy` saved from the script `generate_latents_and_attribute_scores.py`
    input_path: Path = Path("./latents")
    # Where to ave the boundary `npy` files to
    output_path: Path = Path("./boundaries")


@pyrallis.wrap()
def main(opts: TrainConfig):
    all_latent_codes, all_attribute_scores, all_ages, all_poses = [], [], [], []
    for batch_dir in tqdm(opts.input_path.glob("*")):
        if not str(batch_dir.name).startswith("id_"):
            continue
        # load batch latents
        latent_codes = np.load(opts.input_path / batch_dir / 'ws.npy', allow_pickle=True)
        all_latent_codes.extend(latent_codes.tolist())
        # load batch attribute scores
        scores = np.load(opts.input_path / batch_dir / 'scores.npy', allow_pickle=True)
        all_attribute_scores.extend(scores.tolist())
        # load batch ages
        ages = np.load(opts.input_path / batch_dir / 'ages.npy', allow_pickle=True)
        all_ages.extend(ages.tolist())
        # load batch poses
        poses = np.load(opts.input_path / batch_dir / 'poses.npy', allow_pickle=True)
        all_poses.extend(poses.tolist())

    opts.output_path.mkdir(exist_ok=True, parents=True)

    print(f"Obtained a total of {len(all_latent_codes)} latent codes!")

    all_latent_codes = np.array(all_latent_codes)
    all_latent_codes = np.array([l[0] for l in all_latent_codes])

    # train all boundaries for all attributes predicted from the AnyCostGAN classifier
    for attribute_name in attr_list:
        print("Training boundary for: {attribute_name}")
        attr_scores = [s[attr_list.index(attribute_name)][1] for s in all_attribute_scores]
        attr_scores = np.array(attr_scores)[:, np.newaxis]
        boundary = train_boundary(latent_codes=np.array(all_latent_codes),
                                  scores=attr_scores,
                                  chosen_num_or_ratio=0.02,
                                  split_ratio=0.7,
                                  invalid_value=None)
        np.save(opts.output_path / f'{attribute_name}_boundary.npy', boundary)

    # train the age boundary
    boundary = train_boundary(latent_codes=np.array(all_latent_codes),
                              scores=np.array(all_ages),
                              chosen_num_or_ratio=0.02,
                              split_ratio=0.7,
                              invalid_value=None)
    np.save(opts.output_path / f'age_boundary.npy', boundary)

    boundary = train_boundary(latent_codes=np.array(all_latent_codes),
                              scores=np.array(all_poses),
                              chosen_num_or_ratio=0.02,
                              split_ratio=0.7,
                              invalid_value=None)
    np.save(opts.output_path / f'pose_boundary.npy', boundary)


if __name__ == '__main__':
    main()
