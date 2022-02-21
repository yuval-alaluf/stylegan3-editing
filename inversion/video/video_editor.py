from typing import Dict, List, Optional

import torch
from tqdm import tqdm
import numpy as np

from editing.interfacegan.face_editor import FaceEditor
from editing.styleclip_global_directions import edit as styleclip_edit
from inversion.options.train_options import TrainOptions
from utils.fov_expansion import Expander
from inversion.video.post_processing import smooth_ws, smooth_latents_and_transforms, smooth_s
from inversion.video.video_config import VideoConfig
from inversion.video.video_handler import VideoHandler
from models.stylegan3.model import GeneratorType
from models.stylegan3.networks_stylegan3 import Generator
from utils.common import tensor2im, generate_mp4, get_identity_transform


class VideoEditor:

    def __init__(self, generator: Generator, opts: VideoConfig):
        self.opts = opts
        self.generator = generator
        self.expander = Expander(G=self.generator)

    def postprocess_and_smooth_edits(self, results: Dict, edit_latents: np.ndarray, opts: TrainOptions):
        result_latents = np.array(list(results["result_latents"].values()))
        _, smoothed_transforms = smooth_latents_and_transforms(result_latents,
                                                               results["landmarks_transforms"],
                                                               opts=opts)
        if torch.is_tensor(edit_latents[0]):
            edit_latents = np.array([latent.cpu().numpy()[0] for latent in edit_latents])
            edit_latents[:, 9:, :] = edit_latents[:, 9:, :].mean(axis=0)
        print("Generating smoothed edited frames...")
        edited_images_smoothed = self.get_smoothed_edited_images(edit_latents, smoothed_transforms)
        return edited_images_smoothed

    def get_smoothed_edited_images(self, smoothed_edit_latents, smoothed_transforms):
        pass

    def generate_edited_video(self, input_images: List, result_images_smoothed: List[np.ndarray],
                              edited_images_smoothed: List[np.ndarray], video_handler: VideoHandler, save_name: str):
        kwargs = {'fps': video_handler.fps}
        output_path_smoothed = self.opts.output_path / save_name
        generate_mp4(output_path_smoothed, edited_images_smoothed, kwargs)
        coupled_images = []
        for im, smooth_im, edit_im in zip(input_images[2:-2], result_images_smoothed, edited_images_smoothed):
            height, width = smooth_im.shape[:2]
            coupled_im = np.concatenate([im.resize((height, height)), smooth_im, edit_im], axis=1)
            coupled_images.append(coupled_im)
        output_path_coupled = self.opts.output_path / (str(save_name) + "_coupled")
        generate_mp4(output_path_coupled, coupled_images, kwargs)


class InterFaceGANVideoEditor(VideoEditor):

    def __init__(self, generator: Generator, opts: VideoConfig):
        super().__init__(generator=generator, opts=opts)
        self.latent_editor = FaceEditor(stylegan_generator=generator, generator_type=GeneratorType.ALIGNED)

    def edit(self, edit_direction: str, start: int, end: int, result_latents: np.ndarray,
             landmarks_transforms: Optional[np.ndarray] = None):
        print(f"Generating video for {edit_direction} edit...")
        edit_images_start, edit_images_end = [], []
        edit_latents_start, edit_latents_end = [], []
        for latent, landmarks_transform in tqdm(zip(result_latents, landmarks_transforms)):
            if landmarks_transform is not None:
                landmarks_transform = landmarks_transform.cpu().numpy()
                apply_user_transformation = landmarks_transform is not None
                # save the leftmost image in the given range
                res_image, res_latent = self.latent_editor.edit(latents=torch.from_numpy(latent).cuda().unsqueeze(0),
                                                                direction=edit_direction,
                                                                factor=start,
                                                                apply_user_transformations=apply_user_transformation,
                                                                user_transforms=landmarks_transforms)
                edit_images_start.append(res_image)
                edit_latents_start.append(res_latent)
                # save the rightmost image in the given range
                res_image, res_latent = self.latent_editor.edit(latents=torch.from_numpy(latent).cuda().unsqueeze(0),
                                                                direction=edit_direction,
                                                                factor=end,
                                                                apply_user_transformations=apply_user_transformation,
                                                                user_transforms=landmarks_transforms)
                edit_images_end.append(res_image)
                edit_latents_end.append(res_latent)

        # save latents
        latents_path = self.opts.output_path / 'latents'
        latents_path.mkdir(exist_ok=True, parents=True)
        np.save(latents_path / f'latents_{edit_direction}_start.npy', edit_latents_start)
        np.save(latents_path / f'latents_{edit_direction}_end.npy', edit_latents_end)
        return edit_images_start, edit_images_end, edit_latents_start, edit_latents_end

    def get_smoothed_edited_images(self, edit_latents: np.ndarray, smoothed_transforms: torch.tensor):
        smoothed_edit_latents = smooth_ws(edit_latents)
        smoothed_edit_latents = torch.from_numpy(smoothed_edit_latents).float().cuda()
        edited_images_smoothed = []
        for latent, trans in tqdm(zip(smoothed_edit_latents, smoothed_transforms)):
            with torch.no_grad():
                if trans is None:
                    trans = get_identity_transform()
                edited_image = self.expander.generate_expanded_image(ws=latent.unsqueeze(0),
                                                                     landmark_t=trans.cpu().numpy(),
                                                                     pixels_left=self.opts.expansion_amounts[0],
                                                                     pixels_right=self.opts.expansion_amounts[1],
                                                                     pixels_top=self.opts.expansion_amounts[2],
                                                                     pixels_bottom=self.opts.expansion_amounts[3])
            edited_images_smoothed.append(np.array(tensor2im(edited_image[0])))
        return edited_images_smoothed


class StyleCLIPVideoEditor(VideoEditor):

    def __init__(self, generator: Generator, opts: VideoConfig):
        super().__init__(generator=generator, opts=opts)
        styleclip_args = styleclip_edit.EditConfig()
        self.global_direction_calculator = styleclip_edit.load_direction_calculator(generator, opts=styleclip_args)

    def edit(self, edit_direction: str, alpha: float, beta: float, result_latents: np.ndarray,
             landmarks_transforms: Optional[np.ndarray] = None):
        print(f"Generating video for {edit_direction} edit...")
        opts = self._set_opts(alpha, beta, edit_direction)
        edit_results = []
        edit_latents = []
        for idx, (batch_latents, transforms) in tqdm(enumerate(zip(result_latents, landmarks_transforms))):
            batch_transforms = transforms.cpu().numpy()
            edit_res, edit_latent = styleclip_edit.edit_image(latent=batch_latents,
                                                              landmarks_transform=batch_transforms,
                                                              stylegan_model=self.generator,
                                                              global_direction_calculator=self.global_direction_calculator,
                                                              opts=opts,
                                                              image_name=None,
                                                              save=False)
            edit_results.append(tensor2im(edit_res[0]))
            edit_latents.append(edit_latent)

        # save latents
        latents_path = self.opts.output_path / 'latents'
        latents_path.mkdir(exist_ok=True, parents=True)
        save_name = f'result_video_{"_".join(opts.target_text.split())}_{alpha}_{beta}'
        np.save(latents_path / f'latents_{save_name}.npy', edit_latents)
        return edit_results, edit_latents

    def get_smoothed_edited_images(self, edit_latents: List, smoothed_transforms: torch.tensor):
        smoothed_edit_latents = smooth_s([latent[0] for latent in edit_latents])
        edited_images_smoothed = []
        for latent, trans in tqdm(zip(smoothed_edit_latents, smoothed_transforms)):
            with torch.no_grad():
                if trans is None:
                    trans = get_identity_transform()
                edited_image = self.expander.generate_expanded_image(all_s=latent,
                                                                     landmark_t=trans.cpu().numpy(),
                                                                     pixels_left=self.opts.expansion_amounts[0],
                                                                     pixels_right=self.opts.expansion_amounts[1],
                                                                     pixels_top=self.opts.expansion_amounts[2],
                                                                     pixels_bottom=self.opts.expansion_amounts[3])
            edited_images_smoothed.append(np.array(tensor2im(edited_image[0])))
        return edited_images_smoothed

    @staticmethod
    def _set_opts(alpha: float, beta: float, edit_direction: str):
        opts = styleclip_edit.EditConfig()
        opts.alpha_min = alpha
        opts.alpha_max = alpha
        opts.num_alphas = 1
        opts.beta_min = beta
        opts.beta_max = beta
        opts.num_betas = 1
        opts.neutral_text = "a face"
        opts.target_text = edit_direction
        return opts
