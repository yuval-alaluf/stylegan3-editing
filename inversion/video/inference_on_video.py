import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pyrallis
import torch
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from inversion.options.train_options import TrainOptions
from inversion.video.generate_videos import generate_reconstruction_videos
from prepare_data.landmarks_handler import LandmarksHandler
from inversion.video.post_processing import postprocess_and_smooth_inversions
from inversion.video.video_config import VideoConfig
from inversion.video.video_editor import InterFaceGANVideoEditor, StyleCLIPVideoEditor
from inversion.video.video_handler import VideoHandler
from utils.common import tensor2im
from utils.inference_utils import get_average_image, run_on_batch, load_encoder, IMAGE_TRANSFORMS


@pyrallis.wrap()
def run_inference_on_video(video_opts: VideoConfig):
    # prepare all the output paths
    video_opts.output_path.mkdir(exist_ok=True, parents=True)

    # parse video
    video_handler = VideoHandler(video_path=video_opts.video_path,
                                output_path=video_opts.output_path,
                                raw_frames_path=video_opts.raw_frames_path,
                                aligned_frames_path=video_opts.aligned_frames_path,
                                cropped_frames_path=video_opts.cropped_frames_path)
    video_handler.parse_video()

    aligned_paths, cropped_paths = video_handler.get_input_paths()
    input_images = video_handler.load_images(aligned_paths)
    cropped_images = video_handler.load_images(cropped_paths)
    if video_opts.max_images is not None:
        aligned_paths = aligned_paths[:video_opts.max_images]
        input_images = input_images[:video_opts.max_images]
        cropped_images = cropped_images[:video_opts.max_images]

    # load pretrained encoder
    net, opts = load_encoder(video_opts.checkpoint_path, test_opts=video_opts, generator_path=video_opts.generator_path)

    # loads/computes landmarks transforms for the video frames
    landmarks_handler = LandmarksHandler(output_path=video_opts.output_path,
                                         landmarks_transforms_path=video_opts.landmarks_transforms_path)
    video_opts.landmarks_transforms_path = landmarks_handler.landmarks_transforms_path
    landmarks_transforms = landmarks_handler.get_landmarks_transforms(input_paths=aligned_paths,
                                                                      cropped_frames_path=video_handler.cropped_frames_path,
                                                                      aligned_frames_path=video_handler.aligned_frames_path)

    # run inference
    results = run_inference(input_paths=aligned_paths,
                            input_images=input_images,
                            landmarks_transforms=landmarks_transforms,
                            net=net,
                            opts=opts)

    # save inverted latents (can be used for editing, pti, etc)
    results_latents_path = opts.output_path / "latents.npy"
    np.save(results_latents_path, np.array(results["result_latents"]))

    result_images = [np.array(tensor2im(im)) for im in results["result_images"]]
    result_latents = np.array(list(results["result_latents"].values()))
    landmarks_transforms = np.array(list(results["landmarks_transforms"]))

    result_images_smoothed = postprocess_and_smooth_inversions(results, net, video_opts)

    # get video reconstruction
    generate_reconstruction_videos(input_images=cropped_images,
                                   result_images=result_images,
                                   result_images_smoothed=result_images_smoothed,
                                   video_handler=video_handler,
                                   opts=video_opts)

    if opts.interfacegan_directions is not None:
        editor = InterFaceGANVideoEditor(generator=net.decoder, opts=video_opts)
        for interfacegan_edit in video_opts.interfacegan_edits:
            edit_images_start, edit_images_end, edit_latents_start, edit_latents_end = editor.edit(
                edit_direction=interfacegan_edit.direction,
                start=interfacegan_edit.start,
                end=interfacegan_edit.end,
                result_latents=result_latents,
                landmarks_transforms=landmarks_transforms
            )
            edited_images_start_smoothed = editor.postprocess_and_smooth_edits(results, edit_latents_start, video_opts)
            edited_images_end_smoothed = editor.postprocess_and_smooth_edits(results, edit_latents_end, video_opts)
            editor.generate_edited_video(input_images=cropped_images,
                                         result_images_smoothed=result_images_smoothed,
                                         edited_images_smoothed=edited_images_start_smoothed,
                                         video_handler=video_handler,
                                         save_name=f"edited_video_{interfacegan_edit.direction}_start")
            editor.generate_edited_video(input_images=cropped_images,
                                         result_images_smoothed=result_images_smoothed,
                                         edited_images_smoothed=edited_images_end_smoothed,
                                         video_handler=video_handler,
                                         save_name=f"edited_video_{interfacegan_edit.direction}_end")

    if opts.styleclip_directions is not None:
        editor = StyleCLIPVideoEditor(generator=net.decoder, opts=video_opts)
        for styleclip_edit in video_opts.styleclip_edits:
            edited_images, edited_latents = editor.edit(edit_direction=styleclip_edit.target_text,
                                                        alpha=styleclip_edit.alpha,
                                                        beta=styleclip_edit.beta,
                                                        result_latents=result_latents,
                                                        landmarks_transforms=landmarks_transforms)
            edited_images_smoothed = editor.postprocess_and_smooth_edits(results, edited_latents, video_opts)
            editor.generate_edited_video(input_images=cropped_images,
                                         result_images_smoothed=result_images_smoothed,
                                         edited_images_smoothed=edited_images_smoothed,
                                         video_handler=video_handler,
                                         save_name=styleclip_edit.save_name)


def run_inference(input_paths: List[Path], input_images: List, landmarks_transforms: Dict[str, Any], net,
                  opts: TrainOptions):
    results = {"source_images": [], "result_images": [], "result_latents": {}, "landmarks_transforms": []}
    with torch.no_grad():
        avg_image = get_average_image(net)
    # run inference one frame at a time (technically can be run in batches, but done for simplicity)
    for input_image, input_path in tqdm(zip(input_images, input_paths)):
        results["source_images"].append(input_image)
        image_name = input_path.name
        if landmarks_transforms is not None:
            if image_name not in landmarks_transforms:
                continue
            image_landmarks_transform = torch.from_numpy(landmarks_transforms[image_name][-1]).cuda()
        else:
            image_landmarks_transform = None
        with torch.no_grad():
            transformed_image = IMAGE_TRANSFORMS(input_image)
            result_batch, latents = run_on_batch(inputs=transformed_image.unsqueeze(0).cuda(),
                                                 net=net,
                                                 opts=opts,
                                                 avg_image=avg_image,
                                                 landmarks_transform=image_landmarks_transform)
            # we'll save the last inversion and latent code
            results["result_images"].append(result_batch[0][-1])
            results["result_latents"][image_name] = latents[0][-1]
            results["landmarks_transforms"].append(image_landmarks_transform)
    return results


if __name__ == '__main__':
    run_inference_on_video()
