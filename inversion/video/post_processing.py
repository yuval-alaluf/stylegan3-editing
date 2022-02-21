from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm

from utils.fov_expansion import Expander
from inversion.video.video_config import VideoConfig
from utils.common import tensor2im, get_identity_transform


def postprocess_and_smooth_inversions(results: Dict, net, opts: VideoConfig):
    result_latents = np.array(list(results["result_latents"].values()))
    # average fine layers
    result_latents[:, 9:, :] = result_latents[:, 9:, :].mean(axis=0)
    # smooth latents and landmarks transforms
    smoothed_latents, smoothed_transforms = smooth_latents_and_transforms(result_latents,
                                                                          results["landmarks_transforms"],
                                                                          opts=opts)
    # generate the smoothed video frames
    result_images_smoothed = []
    expander = Expander(G=net.decoder)
    print("Generating smoothed frames...")
    for latent, trans in tqdm(zip(smoothed_latents, smoothed_transforms)):
        with torch.no_grad():
            if trans is None:
                trans = get_identity_transform()
            im = expander.generate_expanded_image(ws=latent.unsqueeze(0),
                                                  landmark_t=trans.cpu().numpy(),
                                                  pixels_left=opts.expansion_amounts[0],
                                                  pixels_right=opts.expansion_amounts[1],
                                                  pixels_top=opts.expansion_amounts[2],
                                                  pixels_bottom=opts.expansion_amounts[3])
        result_images_smoothed.append(np.array(tensor2im(im[0])))
    return result_images_smoothed


def smooth_latents_and_transforms(result_latents: np.ndarray, result_landmarks_transforms: List[torch.tensor],
                                  opts: VideoConfig):
    smoothed_latents = smooth_ws(result_latents)
    smoothed_latents = torch.from_numpy(smoothed_latents).float().cuda()
    if opts.landmarks_transforms_path is not None:
        smoothed_transforms = smooth_ws(torch.cat([t.unsqueeze(0) for t in result_landmarks_transforms]))
    else:
        smoothed_transforms = [None] * len(smoothed_latents)
    return smoothed_latents, smoothed_transforms


def smooth_ws(ws: np.ndarray):
    ws_p = ws[2:-2] + 0.75 * ws[3:-1] + 0.75 * ws[1:-3] + 0.25 * ws[:-4] + 0.25 * ws[4:]
    ws_p = ws_p / 3
    return ws_p


def smooth_s(s):
    batched_s = {}
    for c in s[0]:
        bathced_c = torch.cat([s[i][c] for i in range(len(s))])
        batched_s[c] = bathced_c
    new_s = {}
    for c in batched_s:
        new_s[c] = smooth_ws(batched_s[c])
    new_smooth_s = []
    for i in range(new_s['input'].shape[0]):
        curr_s = {c: new_s[c][i].unsqueeze(0) for c in new_s}
        new_smooth_s.append(curr_s)
    return new_smooth_s
