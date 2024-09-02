import numpy as np
from utils.image_process import get_categories, set_categories_img
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from typing import Optional
import torch

from torch.nn import functional as nnf
from einops import rearrange

def cluster_on_patch(scale_h, scale_w, clusters, labels_2d, height_new, width_new, top, left):
    downsize_rgb = np.zeros((height_new, width_new, 3)).astype(np.uint8)
    for i in range(height_new):
        for j in range(width_new):
            labels_count = np.zeros((clusters.shape[0]))
            for k_i in range(scale_h):
                for k_j in range(scale_w):
                    index = labels_2d[top+i*scale_h+k_i][left + j*scale_w + k_j]
                    labels_count[index] +=1    
            chosen_label = labels_count.argmax()
            downsize_rgb[i][j] = clusters[chosen_label]
    return downsize_rgb


#  Function to find the closest points in b for each point in a (2D case)
def closest_points_2d(a, b):
    distances = np.sum(np.square(a[:, np.newaxis] - b), axis=2)
    closest_indices = np.argmin(distances, axis=1)
    return closest_indices

# Function to find the closest points in b for each point in a (3D case)
def closest_points_3d(a, b):
    distances = np.sum(np.square(a[:, :, np.newaxis] - b), axis=3)
    closest_indices = np.argmin(distances, axis=2)
    return closest_indices

def get_closest_colors(input : np.ndarray, palette_colors: np.ndarray):
    # get the label2d with the palette. The input and the palette might not be in RGB color space.
    # The input must be in the same color space as the palette!!! (RGB int != RGB float)
    shape = input.shape
    shape_len = len(shape)
    input = input.astype(np.float32)
    palette_colors = palette_colors.astype(np.float32)
    if shape_len == 2:
        return closest_points_2d(input, palette_colors)
    else: #shape_len == 3
        return closest_points_3d(input, palette_colors)


### Downsampling methods

############
#### Nearest
############

def nearest_downsample(img: torch.Tensor, target_height, target_width):
    downsampled_img = nnf.interpolate(img, size=(target_height, target_width), mode='nearest-exact')
    img_np = downsampled_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)  # Assuming the tensor is in [0, 1]
    colors_np = get_categories(img_np)
    img_cat_np = set_categories_img(img_np, colors_np) # size: (height, width)
    colors = torch.from_numpy(colors_np).to(downsampled_img.device).byte()
    img_cat = torch.from_numpy(img_cat_np).to(downsampled_img.device).long()
    return downsampled_img, img_cat, colors

                           
############
#### Kmeans
############
def kmeans_downsample(img: torch.Tensor, target_height, target_width, num_colors: int = 16, n_init = 4):
    img_np = img.squeeze(0).permute(1, 2, 0)
    height, width, _ = img_np.shape
    img_np = (img_np.cpu().numpy() * 255).astype(np.uint8)  # Assuming the tensor is in [0, 1]
    img_np = rearrange(img_np, 'h w c -> (h w) c')

    # Apply Kmeans
    kmeans = KMeans(n_clusters=num_colors, n_init=n_init).fit(img_np)
    clusters = kmeans.cluster_centers_
    labels = kmeans.labels_
    labels_2d = labels.reshape((height, width))
    scale_h = height // target_height
    scale_w = width // target_width
    top = (height % target_height) // 2
    left = (width % target_width) // 2
    downsize_rgb = cluster_on_patch(scale_h, scale_w, clusters, labels_2d, target_height, target_width, top, left)

    img_cat_np = set_categories_img(downsize_rgb, clusters)
    colors = torch.from_numpy(clusters).to(img.device).byte()
    img_cat = torch.from_numpy(img_cat_np).to(img.device).long()

    downsize_rgb = torch.from_numpy(downsize_rgb).to(img.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    return downsize_rgb, img_cat, colors


############
#### Palette
############
def palette_downsample(img: torch.Tensor, target_height, target_width, palette):
    img_np = img.squeeze(0).permute(1, 2, 0)
    height, width, _ = img_np.shape
    img_np = (img_np.cpu().numpy() * 255).astype(np.uint8)  # Assuming the tensor is in [0, 1]
    img_np = rearrange(img_np, 'h w c -> (h w) c')

    # get the labels here
    labels = get_closest_colors(img_np, palette)
    labels_2d = labels.reshape((height, width))
    scale_h = height // target_height
    scale_w = width // target_width
    top = (height % target_height) // 2
    left = (width % target_width) // 2
    downsize_rgb = cluster_on_patch(scale_h, scale_w, palette, labels_2d, target_height, target_width, top, left)

    img_cat_np = set_categories_img(downsize_rgb, palette)
    colors = torch.from_numpy(palette).to(img.device).byte()
    img_cat = torch.from_numpy(img_cat_np).to(img.device).long()

    downsize_rgb = torch.from_numpy(downsize_rgb).to(img.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return downsize_rgb, img_cat, colors
