import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyrallis
import torch
import torchvision.transforms as transforms
from dataclasses import dataclass
from pyrallis import field
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from criteria.lpips.lpips import LPIPS
from criteria.ms_ssim import MSSSIM
from inversion.datasets.gt_res_dataset import GTResDataset


@dataclass
class RunConfig:
	# Path to reconstructed images
	output_path: Path
	# Path to gt images
	gt_path: Path
	# List of metrics to compute
	metrics: List[str] = field(default=["lpips", "l2", "msssim"], is_mutable=True)
	# Number of works for dataloader
	workers: int = 4
	# Batch size for computing losses
	batch_size: int = 4
	# Stores current metric
	metric: Optional[str] = None


@pyrallis.wrap()
def run(opts: RunConfig):
	for metric in opts.metrics:
		opts.metric = metric
		for step in sorted(opts.output_path.iterdir()):
			if not str(step.name).isdigit():
				continue
			step_outputs_path = opts.output_path / step.name
			if step_outputs_path.is_dir():
				print('#' * 80)
				print(f'Computing {opts.metric} on step: {step.name}')
				print('#' * 80)
				run_on_step_output(step=step.name, opts=opts)


def run_on_step_output(step: str, opts: RunConfig):

	transform = transforms.Compose([transforms.Resize((256, 256)),
									transforms.ToTensor(),
									transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

	step_outputs_path = opts.output_path / step

	print('Loading dataset')
	dataset = GTResDataset(root_path=step_outputs_path,
						   gt_dir=opts.gt_path,
						   transform=transform)

	dataloader = DataLoader(dataset,
							batch_size=opts.batch_size,
							shuffle=False,
							num_workers=int(opts.workers),
							drop_last=True)

	if opts.metric == 'lpips':
		loss_func = LPIPS(net_type='alex')
	elif opts.metric == 'l2':
		loss_func = torch.nn.MSELoss()
	elif opts.metric == 'msssim':
		loss_func = MSSSIM()
	else:
		raise Exception(f'Not a valid metric: {opts.metric}!')

	loss_func.cuda()

	global_i = 0
	scores_dict = {}
	all_scores = []
	for result_batch, gt_batch in tqdm(dataloader):
		for i in range(opts.batch_size):
			loss = float(loss_func(result_batch[i:i+1].cuda(), gt_batch[i:i+1].cuda()))
			all_scores.append(loss)
			im_path = dataset.pairs[global_i][0]
			scores_dict[im_path.name] = loss
			global_i += 1

	all_scores = list(scores_dict.values())
	mean = np.mean(all_scores)
	std = np.std(all_scores)
	result_str = f'Average loss is {mean:.2f}+-{std:.2f}'
	print('Finished with ', step_outputs_path)
	print(result_str)

	out_path = opts.output_path.parent / 'inference_metrics'
	out_path.mkdir(exist_ok=True, parents=True)

	with open(out_path / f'stat_{opts.metric}_step_{step}.txt', 'w') as f:
		f.write(result_str)
	with open(out_path / f'scores_{opts.metric}_step_{step}.json', 'w') as f:
		json.dump(scores_dict, f)


if __name__ == '__main__':
	run()
