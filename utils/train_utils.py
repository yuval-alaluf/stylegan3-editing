from typing import Dict, List, Any

import matplotlib.pyplot as plt


def aggregate_loss_dict(agg_loss_dict):
	mean_vals = {}
	for output in agg_loss_dict:
		for key in output:
			mean_vals[key] = mean_vals.setdefault(key, []) + [output[key]]
	for key in mean_vals:
		if len(mean_vals[key]) > 0:
			mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
		else:
			print(f'{key} has no value')
			mean_vals[key] = 0
	return mean_vals


def vis_faces(log_hooks: List[Dict]):
	display_count = len(log_hooks)
	n_outputs = len(log_hooks[0]['output_face']) if type(log_hooks[0]['output_face']) == list else 1
	fig = plt.figure(figsize=(6 + (n_outputs * 2), 4 * display_count))
	gs = fig.add_gridspec(display_count, (2 + n_outputs))
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		vis_faces_iterative(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig


def vis_faces_iterative(hooks_dict: Dict[str, Any], fig, gs, i: int):
	plt.imshow(hooks_dict['input_face'])
	plt.title(f'Input\nOut Sim={float(hooks_dict["diff_input"]):.2f}')
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title(f'Target\nIn={float(hooks_dict["diff_views"]):.2f}, Out={float(hooks_dict["diff_target"]):.2f}')
	for idx, output_idx in enumerate(range(len(hooks_dict['output_face']) - 1, -1, -1)):
		output_image, similarity = hooks_dict['output_face'][output_idx]
		fig.add_subplot(gs[i, 2 + idx])
		plt.imshow(output_image)
		plt.title(f'Output {output_idx}\n Target Sim={float(similarity):.2f}')
