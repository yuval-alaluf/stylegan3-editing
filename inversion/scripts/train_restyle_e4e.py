import json
import pprint
import sys
from plistlib import Dict
from typing import Any

import dataclasses
import pyrallis
import torch

sys.path.append(".")
sys.path.append("..")

from inversion.options.e4e_train_options import e4eTrainOptions
from inversion.training.coach_restyle_e4e import Coach


@pyrallis.wrap()
def main(opts: e4eTrainOptions):
	previous_train_ckpt = None
	if opts.resume_training_from_ckpt:
		opts, previous_train_ckpt = load_train_checkpoint(opts)
	else:
		setup_progressive_steps(opts)
		create_initial_experiment_dir(opts)

	coach = Coach(opts, previous_train_ckpt)
	coach.train()


def load_train_checkpoint(opts: e4eTrainOptions):
	train_ckpt_path = opts.resume_training_from_ckpt
	previous_train_ckpt = torch.load(opts.resume_training_from_ckpt, map_location='cpu')
	new_opts_dict = dataclasses.asdict(opts)
	opts = previous_train_ckpt['opts']
	opts['resume_training_from_ckpt'] = train_ckpt_path
	update_new_configs(opts, new_opts_dict)
	pprint.pprint(opts)
	opts = e4eTrainOptions(**opts)
	if opts.sub_exp_dir is not None:
		sub_exp_dir = opts.sub_exp_dir
		opts.exp_dir = opts.exp_dir / sub_exp_dir
	create_initial_experiment_dir(opts)
	return opts, previous_train_ckpt


def setup_progressive_steps(opts: e4eTrainOptions):
	num_style_layers = 16
	num_deltas = num_style_layers - 1
	if opts.progressive_start is not None:  # If progressive delta training
		opts.progressive_steps = [0]
		next_progressive_step = opts.progressive_start
		for i in range(num_deltas):
			opts.progressive_steps.append(next_progressive_step)
			next_progressive_step += opts.progressive_step_every

	assert opts.progressive_steps is None or is_valid_progressive_steps(opts, num_style_layers), \
		"Invalid progressive training input"


def is_valid_progressive_steps(opts: e4eTrainOptions, num_style_layers: int):
	return len(opts.progressive_steps) == num_style_layers and opts.progressive_steps[0] == 0


def create_initial_experiment_dir(opts: e4eTrainOptions):
	opts.exp_dir.mkdir(exist_ok=True, parents=True)
	opts_dict = dataclasses.asdict(opts)
	pprint.pprint(opts_dict)
	with open(opts.exp_dir / 'opt.json', 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True, default=str)


def update_new_configs(ckpt_opts: Dict[str, Any], new_opts: Dict[str, Any]):
	for k, v in new_opts.items():
		ckpt_opts[k] = v


if __name__ == '__main__':
	main()
