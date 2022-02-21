import os

import dlib
import numpy as np
import scipy.ndimage
from PIL import Image

from utils.common import make_transform


def get_landmark(filepath, detector, predictor):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)

    for k, d in enumerate(dets):
        shape = predictor(img, d)

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm


def get_eyes_coors(landmark):
    lm_eye_left = landmark[36: 42]  # left-clockwise
    lm_eye_right = landmark[42: 48]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)

    return eye_left, eye_right


def get_rotation_from_eyes(left_eye_unaligned, right_eye_unaligned, left_eye_aligned, right_eye_aligned):
    eye_to_eye1 = right_eye_unaligned - left_eye_unaligned
    eye_to_eye_normalized1 = eye_to_eye1 / np.linalg.norm(eye_to_eye1)
    eye_to_eye2 = right_eye_aligned - left_eye_aligned
    eye_to_eye_normalized2 = eye_to_eye2 / np.linalg.norm(eye_to_eye2)

    cos_r = np.inner(eye_to_eye_normalized1, eye_to_eye_normalized2)
    r_rad = np.arccos(cos_r)
    r = np.degrees(r_rad)
    if right_eye_unaligned[1] > left_eye_unaligned[1]:
        r = 360 - r
    return r


def get_alignment_positions(filepath: str, detector, predictor, eyes_distance_only: bool = True):
    lm = get_landmark(filepath, detector, predictor)

    lm_mouth_outer = lm[48: 60]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left, eye_right = get_eyes_coors(lm)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    if eyes_distance_only:
        x *= np.hypot(*eye_to_eye) * 2.0
    else:
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1

    return c, x, y


def get_alignment_transformation(c: np.ndarray, x: np.ndarray, y: np.ndarray):
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2
    return quad, qsize


def get_fixed_cropping_transformation(c, x):
    d = np.hypot(x[0], x[1])
    d_hor = np.array([d, 0])
    d_ver = np.array([0, d])
    quad = np.stack([c - d_hor - d_ver, c - d_hor + d_ver, c + d_hor + d_ver, c + d_hor - d_ver])
    qsize = np.hypot(*x) * 2

    return quad, qsize


def crop_face_by_transform(filepath: str, quad: np.ndarray, qsize: int, output_size: int = 1024,
                           transform_size: int = 1024, enable_padding: bool = True):
    # read image
    img = Image.open(filepath)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    # Return aligned image.
    return img


def align_face(filepath: str, detector, predictor):
    c, x, y = get_alignment_positions(filepath, detector, predictor)
    quad, qsize = get_alignment_transformation(c, x, y)
    img = crop_face_by_transform(filepath, quad, qsize)
    return img


def crop_face(filepath: str, detector, predictor, random_shift=0.0):
    c, x, y = get_alignment_positions(filepath, detector, predictor)
    if random_shift > 0:
        c = (c + np.hypot(*x) * 2 * random_shift * np.random.normal(0, 1, c.shape))
    quad, qsize = get_fixed_cropping_transformation(c, x)
    img = crop_face_by_transform(filepath, quad, qsize)
    return img


def get_stylegan_transform(unaligned_path: str, aligned_path: str, detector, predictor):
    try:
        # if the aligned image already exists, we can skip the prepare_data part to save time
        if not os.path.exists(aligned_path):
            aligned_img = align_face(unaligned_path, detector, predictor)
            aligned_img.save(aligned_path)
        else:
            aligned_img = Image.open(aligned_path).convert("RGB")

        aligned_img_lm = get_landmark(aligned_path, detector, predictor)
        aligned_left_eye, aligned_right_eye = get_eyes_coors(aligned_img_lm)
        unaligned_img_lm = get_landmark(unaligned_path, detector, predictor)
        unaligned_left_eye, unaligned_right_eye = get_eyes_coors(unaligned_img_lm)

        rotation_angle = get_rotation_from_eyes(left_eye_unaligned=unaligned_left_eye,
                                                right_eye_unaligned=unaligned_right_eye,
                                                left_eye_aligned=aligned_left_eye,
                                                right_eye_aligned=aligned_right_eye)

        rotated_aligned_image = Image.open(aligned_path).rotate(rotation_angle)
        unaligned_name = os.path.basename(unaligned_path).split('.')[0]
        rotated_aligned_image.save(f"rotated_aligned_image_{unaligned_name}.png")

        rotated_aligned_img_lm = get_landmark(f"rotated_aligned_image_{unaligned_name}.png", detector, predictor)
        rotated_aligned_left_eye, rotated_aligned_right_eye = get_eyes_coors(rotated_aligned_img_lm)

        os.remove(f"rotated_aligned_image_{unaligned_name}.png")

        translation = (unaligned_left_eye - rotated_aligned_left_eye) / aligned_img.size[0]

        transform = make_transform(translation, rotation_angle)
        inverse_transform = np.linalg.inv(transform)

        return rotation_angle, translation, transform, inverse_transform

    except Exception as e:
        print(f"Failed aligning image: {aligned_path}. Got exception: {e}")
        return None
