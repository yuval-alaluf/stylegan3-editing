import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pyrallis
import torch
import torchvision.transforms as trans
from PIL import Image
from dataclasses import dataclass

sys.path.append(".")
sys.path.append("..")

from inversion.models.mtcnn.mtcnn import MTCNN
from inversion.models.encoders.model_irse import IR_101
from configs.paths_config import model_paths


CIRCULAR_FACE_PATH = model_paths['curricular_face']


@dataclass
class RunConfig:
	# Path to reconstructed images
	output_path: Path
	# Path to gt images
	gt_path: Path
	# Number of works to use for computing losses in parallel
	num_threads: int = 4


@pyrallis.wrap()
def run(opts: RunConfig):
	for step in sorted(opts.output_path.glob("*")):
		if not str(step.name).isdigit():
			continue
		step_outputs_path = opts.output_path / step.name
		if step_outputs_path.is_dir():
			print('#' * 80)
			print(f'Running on step: {step.name}')
			print('#' * 80)
			run_on_step_output(step=step.name, args=opts)


def run_on_step_output(step: str, args: RunConfig):
	file_paths = []
	step_outputs_path = args.output_path / step
	for f in step_outputs_path.glob("*"):
		image_path = step_outputs_path / f
		gt_path = args.gt_path / f
		if f.suffix in [".jpg", ".png", ".jpeg"]:
			file_paths.append([image_path, gt_path])

	file_chunks = list(chunks(file_paths, int(math.ceil(len(file_paths) / args.num_threads))))
	pool = mp.Pool(args.num_threads)
	print(f'Running on {len(file_paths)} paths\nHere we goooo')

	tic = time.time()
	results = pool.map(extract_on_paths, file_chunks)
	scores_dict = {}
	for d in results:
		scores_dict.update(d)

	all_scores = list(scores_dict.values())
	mean = np.mean(all_scores)
	std = np.std(all_scores)
	result_str = f'New Average score is {mean:.2f}+-{std:.2f}'
	print(result_str)

	out_path = args.output_path.parent / 'inference_metrics'
	out_path.mkdir(exist_ok=True, parents=True)

	with open(out_path / f'stat_id_step_{step}.txt', 'w') as f:
		f.write(result_str)
	with open(out_path / f'scores_id_step_{step}.json', 'w') as f:
		json.dump(scores_dict, f)

	toc = time.time()
	print(f'Mischief managed in {tic - toc}s')


def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]


def extract_on_paths(file_paths):
	facenet = IR_101(input_size=112)
	facenet.load_state_dict(torch.load(CIRCULAR_FACE_PATH))
	facenet.cuda()
	facenet.eval()
	mtcnn = MTCNN()
	id_transform = trans.Compose([
		trans.ToTensor(),
		trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	])

	pid = mp.current_process().name
	print(f'\t{pid} is starting to extract on {len(file_paths)} images')
	tot_count = len(file_paths)
	count = 0

	scores_dict = {}
	for res_path, gt_path in file_paths:
		count += 1
		if count % 100 == 0:
			print(f'{pid} done with {count}/{tot_count}')
		if True:
			input_im = Image.open(res_path)
			input_im, _ = mtcnn.align(input_im)
			if input_im is None:
				print(f'{pid} skipping {res_path}')
				continue

			input_id = facenet(id_transform(input_im).unsqueeze(0).cuda())[0]

			result_im = Image.open(gt_path)
			result_im, _ = mtcnn.align(result_im)
			if result_im is None:
				print(f'{pid} skipping {gt_path}')
				continue

			result_id = facenet(id_transform(result_im).unsqueeze(0).cuda())[0]
			score = float(input_id.dot(result_id))
			scores_dict[gt_path.name] = score

	return scores_dict


if __name__ == '__main__':
	run()
