import torch
import numpy as np

from models.stylegan3.networks_stylegan3 import Generator
from utils.common import make_transform


class Expander:

    def __init__(self, G: Generator):
        self.G = G

    def generate_expanded_image(self, ws=None, all_s=None, landmark_t=None,
                                pixels_right=0, pixels_left=0, pixels_top=0, pixels_bottom=0):
        assert landmark_t is not None, "Expected to receive landmarks transforms! Received None!"

        images = []
        transforms = Expander._get_transforms(self.G.img_resolution, pixels_right, pixels_left, pixels_top, pixels_bottom)

        for t in transforms:
            if t is not None:
                self.G.synthesis.input.transform = torch.from_numpy(landmark_t @ t).float().cuda()
                with torch.no_grad():
                    img = self.G.synthesis(ws, all_s)
            else:
                img = None
            images.append(img)

        merged_image = Expander._merge_images(images, self.G.img_resolution, pixels_right, pixels_left, pixels_top, pixels_bottom)
        return merged_image


    @staticmethod
    def _get_transforms(res, pixels_right, pixels_left, pixels_top, pixels_bottom):
        identity_transform = make_transform((0, 0), 0)
        transform_left = Expander._get_transform_single_edge(res, edge="left", num_pixels=pixels_left)
        transform_right = Expander._get_transform_single_edge(res, edge="right", num_pixels=pixels_right)
        transform_top = Expander._get_transform_single_edge(res, edge="top", num_pixels=pixels_top)
        transform_bottom = Expander._get_transform_single_edge(res, edge="bottom", num_pixels=pixels_bottom)

        transform_top_left = Expander._get_transform_corner(res, corner="top_left", num_pixels_hor=pixels_left, num_pixels_ver=pixels_top)
        transform_top_right = Expander._get_transform_corner(res, corner="top_right", num_pixels_hor=pixels_right, num_pixels_ver=pixels_top)
        transform_bottom_left = Expander._get_transform_corner(res, corner="bottom_left", num_pixels_hor=pixels_left, num_pixels_ver=pixels_bottom)
        transform_bottom_right = Expander._get_transform_corner(res, corner="bottom_right", num_pixels_hor=pixels_right, num_pixels_ver=pixels_bottom)

        transforms = [identity_transform, transform_left, transform_top, transform_right, transform_bottom,
                      transform_top_left, transform_top_right, transform_bottom_right, transform_bottom_left]

        for i in range(len(transforms)):
            if transforms[i] is not None:
                transforms[i] = np.linalg.inv(transforms[i])

        return transforms

    @staticmethod
    def _get_transform_single_edge(res, edge, num_pixels):
        if num_pixels == 0:
            return None
        if edge == "left":
            return make_transform((num_pixels / res, 0), 0)
        if edge == "right":
            return make_transform((-num_pixels / res, 0), 0)
        if edge == "top":
            return make_transform((0, num_pixels / res), 0)
        if edge == "bottom":
            return make_transform((0, -num_pixels / res), 0)
        else:
            raise ValueError("Invalid edge for transform")

    @staticmethod
    def _get_transform_corner(res, corner, num_pixels_hor, num_pixels_ver):
        if num_pixels_hor == 0 or num_pixels_ver == 0:
            return None
        if corner == "top_left":
            return make_transform((num_pixels_hor / res, num_pixels_ver / res), 0)
        if corner == "top_right":
            return make_transform((-num_pixels_hor / res, num_pixels_ver / res), 0)
        if corner == "bottom_left":
            return make_transform((num_pixels_hor / res, -num_pixels_ver / res), 0)
        if corner == "bottom_right":
            return make_transform((-num_pixels_hor / res, -num_pixels_ver / res), 0)
        else:
            raise ValueError("Invalid corner for transform")

    @staticmethod
    def _merge_images(images, res, pixels_right, pixels_left, pixels_top, pixels_bottom):
        result_image = torch.zeros(images[0].shape[0], 3, pixels_top + res + pixels_bottom, pixels_left + res + pixels_right).cuda()
        # center
        result_image[:, :, pixels_top:pixels_top + res, pixels_left:pixels_left + res] = images[0]
        if pixels_left > 0:
            result_image[:, :, pixels_top:pixels_top + res, :pixels_left] = images[1][:, :, :, 0:pixels_left]
        if pixels_top > 0:
            result_image[:, :, :pixels_top, pixels_left:pixels_left + res] = images[2][:, :, 0:pixels_top, :]
        if pixels_right > 0:
            result_image[:, :, pixels_top:pixels_top + res, pixels_left + res:] = images[3][:, :, :, res - pixels_right:]
        if pixels_bottom > 0:
            result_image[:, :, pixels_top + res:, pixels_left:pixels_left + res] = images[4][:, :, res - pixels_bottom:, :]

        if pixels_top > 0 and pixels_left > 0:
            result_image[:, :, :pixels_top, :pixels_left] = images[5][:, :, :pixels_top, :pixels_left]
        if pixels_top > 0 and pixels_right > 0:
            result_image[:, :, :pixels_top, res + pixels_left:] = images[6][:, :, :pixels_top, res - pixels_right:]
        if pixels_bottom > 0 and pixels_right > 0:
            result_image[:, :, res + pixels_top:, res + pixels_left:] = images[7][:, :, res - pixels_bottom:, res - pixels_right:]
        if pixels_bottom > 0 and pixels_left > 0:
            result_image[:, :, res + pixels_top:, :pixels_left] = images[8][:, :, res - pixels_bottom:, :pixels_left]

        return result_image
