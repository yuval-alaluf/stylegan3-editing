import sys
import time
from typing import Optional

import numpy as np
import pyrallis
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from inversion.datasets.inference_dataset import InferenceDataset
from editing.interfacegan.face_editor import FaceEditor
from inversion.options.test_options import TestOptions
from models.stylegan3.model import GeneratorType
from utils.common import tensor2im
from utils.inference_utils import get_average_image, run_on_batch, load_encoder


@pyrallis.wrap()
def run(test_opts: TestOptions):

    out_path_results = test_opts.output_path / 'editing_results'
    out_path_results.mkdir(exist_ok=True, parents=True)

    # update test options with options used during training
    net, opts = load_encoder(checkpoint_path=test_opts.checkpoint_path, test_opts=test_opts)

    print(f'Loading dataset for {opts.dataset_type}')
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               landmarks_transforms_path=opts.landmarks_transforms_path,
                               transform=transforms_dict['transform_inference'])
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    # prepare editing directions and ranges
    latent_editor = FaceEditor(net.decoder, generator_type=GeneratorType.ALIGNED)

    avg_image = get_average_image(net)

    resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)

    global_i = 0
    global_time = []
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            input_batch, landmarks_transform = input_batch
            tic = time.time()
            result_batch = edit_batch(inputs=input_batch.cuda().float(),
                                      net=net,
                                      avg_image=avg_image,
                                      latent_editor=latent_editor,
                                      opts=opts,
                                      landmarks_transform=landmarks_transform.cuda().float())
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(input_batch.shape[0]):

            im_path = dataset.paths[global_i]
            results = result_batch[i]

            inversion = results.pop('inversion')
            input_im = tensor2im(input_batch[i])

            all_edit_results = []
            for edit_name, edit_res in results.items():
                res = np.array(input_im.resize(resize_amount))  # set the input image
                res = np.concatenate([res, np.array(inversion.resize(resize_amount))], axis=1)  # set the inversion
                for result in edit_res:
                    res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)
                res_im = Image.fromarray(res)
                all_edit_results.append(res_im)

                edit_save_dir = out_path_results / edit_name
                edit_save_dir.mkdir(exist_ok=True, parents=True)
                res_im.save(edit_save_dir / im_path.name)

            global_i += 1

    stats_path = opts.output_path / 'stats.txt'
    result_str = f'Runtime {np.mean(global_time):.4f}+-{np.std(global_time):.4f}'
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)


def edit_batch(inputs: torch.tensor, net, avg_image: torch.tensor, latent_editor: FaceEditor, opts: TestOptions,
               landmarks_transform: Optional[torch.tensor] = None):
    y_hat, latents = get_inversions_on_batch(inputs=inputs,
                                             net=net,
                                             avg_image=avg_image,
                                             opts=opts,
                                             landmarks_transform=landmarks_transform)
    # store all results for each sample, split by the edit direction
    results = {idx: {'inversion': tensor2im(y_hat[idx])} for idx in range(len(inputs))}
    for edit_direction, factor_range in zip(opts.edit_directions, opts.factor_ranges):
        edit_res = latent_editor.edit(latents=latents,
                                      direction=edit_direction,
                                      factor_range=factor_range,
                                      apply_user_transformations=True,
                                      user_transforms=landmarks_transform)
        edit_images, _ = edit_res
        # store the results for each sample
        for idx in range(inputs.shape[0]):
            results[idx][edit_direction] = [step_res[idx] for step_res in edit_images]
    return results


def get_inversions_on_batch(inputs: torch.tensor, net, avg_image: torch.tensor, opts: TestOptions,
                            landmarks_transform: Optional[torch.tensor] = None):
    result_batch, result_latents = run_on_batch(inputs=inputs,
                                                net=net,
                                                opts=opts,
                                                avg_image=avg_image,
                                                landmarks_transform=landmarks_transform)
    # we'll take the final inversion as the inversion to edit
    y_hat = [result_batch[idx][-1] for idx in range(len(result_batch))]
    latents = [torch.from_numpy(result_latents[idx][-1]).cuda() for idx in range(len(result_batch))]
    return y_hat, torch.stack(latents)


if __name__ == '__main__':
    run()
