import pathlib
from typing import Union, List, Text, BinaryIO
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as nnf


### Distance functions used for intialization ###
def euclidean_similarity_target_cls(target_cls_index, palette, distance_type='l2'):
    palette_norm = palette.float() / 255.0
    h, w = target_cls_index.shape
    n = palette_norm.shape[0]
    target = torch.zeros((h, w, n), device=palette_norm.device)
    for i in range(h):
        for j in range(w):
            ref_color_idx = target_cls_index[i, j].item()
            ref_color = palette_norm[ref_color_idx].unsqueeze(0)
            # Compute negative distance
            if distance_type == 'l2':
                distance = 1.0-torch.norm(ref_color - palette_norm, dim=1)
            elif distance_type == 'l1':
                distance = 1.0-torch.abs(ref_color - palette_norm).sum(dim=1)
            else:
                raise ValueError("Unsupported distance type")
            target[i, j] = distance
    return target

def euclidean_similarity_target(target_img, palette, distance_type = 'l2'):
    palette_norm = palette.float() / 255.0
    h, w, _ = target_img.shape
    n = palette_norm.shape[0]
    target = torch.zeros((h, w, n), device=palette_norm.device)
    colored_image = torch.zeros_like(target_img, device=palette_norm.device)
    for i in range(h):
        for j in range(w):
            ref_color = target_img[i, j].unsqueeze(0)
            # Compute negative distance
            if distance_type == 'l2':
                distance = -torch.norm(ref_color - palette_norm, dim=1)
            elif distance_type == 'l1':
                distance = -torch.abs(ref_color - palette_norm).sum(dim=1)
            else:
                raise ValueError("Unsupported distance type")
            target[i, j] = distance
            # Find the index of the closest color in the palette
            closest_color_idx = torch.argmax(distance)
            colored_image[i, j] = palette_norm[closest_color_idx]
    return target, colored_image

def cosine_similarity_create_target_cls(target_cls_index, palette):
    palette_norm = palette.float() / 255.0
    palette_norm = palette_norm * 2 - 1
    h, w = target_cls_index.shape
    n = palette_norm.shape[0]
    target = torch.zeros((h, w, n), device=palette_norm.device)
    eps = 1e-2
    # Calculate cosine similarity for each pixel
    for i in range(h):
        for j in range(w):
            ref_color_idx = target_cls_index[i, j].item()
            ref_color = palette_norm[ref_color_idx].unsqueeze(0)
            cos_sim = nnf.cosine_similarity(ref_color, palette_norm, dim=1)
            target[i, j] = cos_sim
            target[i, j, ref_color_idx] = 1.0 + eps #the cosine similarity will not distinguish between colors that are aligned. For instance, greyscale that is all on the same line! By adding a small eps, we make sure that the color is selected.
    return target

###############################################


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                save_image: bool = False,
                fp: Union[Text, pathlib.Path, BinaryIO] = None) -> np.ndarray:
    if save_image:
        assert fp is not None

    if isinstance(images, np.ndarray) and images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images] if not isinstance(images, list) else images
        num_empty = len(images) % num_rows

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    # Calculate the composite image
    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = int(np.ceil(num_items / num_rows))  # count the number of columns
    image_h = h * num_rows + offset * (num_rows - 1)
    image_w = w * num_cols + offset * (num_cols - 1)
    assert image_h > 0, "Invalid image height: {} (num_rows={}, offset_ratio={}, num_items={})".format(
        image_h, num_rows, offset_ratio, num_items)
    assert image_w > 0, "Invalid image width: {} (num_cols={}, offset_ratio={}, num_items={})".format(
        image_w, num_cols, offset_ratio, num_items)
    image_ = np.ones((image_h, image_w, 3), dtype=np.uint8) * 255

    # Ensure that the last row is filled with empty images if necessary
    if len(images) % num_cols > 0:
        empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
        num_empty = num_cols - len(images) % num_cols
        images += [empty_images] * num_empty

    for i in range(num_rows):
        for j in range(num_cols):
            k = i * num_cols + j
            if k >= num_items:
                break
            image_[i * (h + offset): i * (h + offset) + h, j * (w + offset): j * (w + offset) + w] = images[k]

    pil_img = Image.fromarray(image_)
    if save_image:
        pil_img.save(fp)
    return pil_img
