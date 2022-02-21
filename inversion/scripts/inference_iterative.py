import sys
import time

import numpy as np
import pyrallis
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from inversion.options.test_options import TestOptions
from inversion.datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im
from utils.inference_utils import get_average_image, run_on_batch, load_encoder


@pyrallis.wrap()
def run(test_opts: TestOptions):

    out_path_results = test_opts.output_path / 'inference_results'
    out_path_coupled = test_opts.output_path / 'inference_coupled'
    out_path_results.mkdir(exist_ok=True, parents=True)
    out_path_coupled.mkdir(exist_ok=True, parents=True)

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

    # get the image corresponding to the latent average
    avg_image = get_average_image(net)

    resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)

    global_i = 0
    global_time = []
    all_latents = {}
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break

        with torch.no_grad():
            input_batch, landmarks_transform = input_batch
            tic = time.time()
            result_batch, result_latents = run_on_batch(inputs=input_batch.cuda().float(),
                                                        net=net,
                                                        opts=opts,
                                                        avg_image=avg_image,
                                                        landmarks_transform=landmarks_transform.cuda().float())
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(input_batch.shape[0]):
            results = [tensor2im(result_batch[i][iter_idx]) for iter_idx in range(opts.n_iters_per_batch)]
            im_path = dataset.paths[global_i]

            # save individual step results
            for idx, result in enumerate(results):
                save_dir = out_path_results / str(idx)
                save_dir.mkdir(exist_ok=True, parents=True)
                result.resize(resize_amount).save(save_dir / im_path.name)

            # save step-by-step results side-by-side
            input_im = tensor2im(input_batch[i])
            res = np.array(results[0].resize(resize_amount))
            for idx, result in enumerate(results[1:]):
                res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)
            res = np.concatenate([res, input_im.resize(resize_amount)], axis=1)
            Image.fromarray(res).save(out_path_coupled / im_path.name)

            # store all latents with dict pairs (image_name, latents)
            all_latents[im_path.name] = result_latents[i]

            global_i += 1

    stats_path = opts.output_path / 'stats.txt'
    result_str = f'Runtime {np.mean(global_time):.4f}+-{np.std(global_time):.4f}'
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)

    # save all latents as npy file
    np.save(test_opts.output_path / 'latents.npy', all_latents)


if __name__ == '__main__':
    run()
