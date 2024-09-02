from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from PIL import Image
from math import sqrt
import torchvision.transforms.functional as TF

def total_variation_loss(img, method = "l1"):
    """
    Compute total variation loss.

    Args:
        img (Tensor): The image tensor of shape (H, W, C).

    Returns:
        Tensor: Total variation loss.
    """

    if method == "l1":
        tv_h = torch.abs(img[:, :-1, :] - img[:, 1:, :]).mean()
        tv_w = torch.abs(img[:-1, :, :] - img[1:, :, :]).mean()
    elif method == "l2":
        tv_h = torch.pow(img[:, :-1, :] - img[:, 1:, :], 2).mean()
        tv_w = torch.pow(img[:-1, :, :] - img[1:, :, :], 2).mean()
    return tv_h + tv_w

def gradient_loss(img_pix, img_target):
    img_pix = img_pix.permute(2, 0, 1).unsqueeze(0)
    img_pix = F.interpolate(img_pix, size = img_target.shape[2:], mode="bilinear", align_corners=False)

    # Apply Sobel operator for horizontal and vertical gradients on grayscale
    img1 = TF.rgb_to_grayscale(img_pix)
    img2 = TF.rgb_to_grayscale(img_target)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img1.dtype).view(1, 1, 3, 3).to(img1.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img1.dtype).view(1, 1, 3, 3).to(img1.device)
    grad_x1 = F.conv2d(img1, sobel_x, padding=1)
    grad_y1 = F.conv2d(img1, sobel_y, padding=1)
    grad_x2 = F.conv2d(img2, sobel_x, padding=1)
    grad_y2 = F.conv2d(img2, sobel_y, padding=1)

    loss = F.mse_loss(grad_x1, grad_x2) + F.mse_loss(grad_y1, grad_y2)
    return loss, grad_x1, grad_y1, grad_x2, grad_y2

class FFT_Loss:
    def __init__(self, device, height, width) -> None:
        self.center_H = height / 2.0
        self.center_W = width / 2.0
        self.radius = min(height, width) / (4*sqrt(2))
        self.mask = torch.ones((height, width))
        Y, X = torch.meshgrid(torch.arange(height), torch.arange(width))
        center_distance = torch.sqrt((X - self.center_W)**2 + (Y - self.center_H)**2)
        self.mask[center_distance < self.radius] = 0
        self.mask = self.mask.unsqueeze(0).repeat(3, 1, 1)
        self.mask = self.mask.to(device)

    def get_mask(self):
        return self.mask
    
    def get_loss_img(self, img):
        """
        Compute loss based on the Fourier Transform to penalize high-frequency components. Does these 4 steps:
        1- Apply Fast Fourier Transform
        2- Shift the zero-frequency component to the center
        3- Calculate L1 magnitude spectrum
        4- Penalize high-frequency components by returning the loss over the high-magnitude components.

        Args:
            img (Tensor): The image tensor of shape (H, W, C).

        Returns:
            Tensor: Fourier Transform-based loss.
        """
        shifted_img = rearrange(img, 'h w c -> c h w')
        fft_img = torch.fft.fft2(shifted_img)
        fft_img = torch.fft.fftshift(fft_img)
        magnitude = torch.abs(fft_img)
        high_freq_loss = (magnitude * self.mask) / (self.mask.sum() + 1e-8)
        return high_freq_loss, magnitude, self.mask

    
    def __call__(self, img) -> Any:
        """
        Compute loss based on the Fourier Transform to penalize high-frequency components. Does these 4 steps:
        1- Apply Fast Fourier Transform
        2- Shift the zero-frequency component to the center
        3- Calculate L1 magnitude spectrum
        4- Penalize high-frequency components by returning the loss over the high-magnitude components.

        Args:
            img (Tensor): The image tensor of shape (H, W, C).

        Returns:
            Tensor: Fourier Transform-based loss.
        """
        shifted_img = rearrange(img, 'h w c -> c h w')
        fft_img = torch.fft.fft2(shifted_img)
        fft_img = torch.fft.fftshift(fft_img)
        magnitude = torch.abs(fft_img)
        high_freq_loss = (magnitude * self.mask).sum() / (self.mask.sum() + 1e-8)

        return high_freq_loss

def create_distance_matrix(max_distance):
    """ Create a matrix of distances for given max distance. """
    range = torch.arange(-max_distance, max_distance + 1, dtype=torch.float32)
    dist_matrix = torch.sqrt(range[:, None]**2 + range[None, :]**2)
    return dist_matrix

def bilateral_edge_preserving_loss(img, sigma_color, sigma_space, max_distance=2):
    """
    Compute an edge-preserving loss for an image considering multi-directional neighbors.

    Args:
        img (Tensor): The image tensor of shape (C, H, W).
        sigma_color (float): Controls how the function penalizes intensity differences.
        sigma_space (float): Controls the spatial extent of the neighborhood.
        max_distance (int): Maximum distance for considering a pixel as a neighbor.

    Returns:
        Tensor: Edge-preserving loss.
    """
    H, W, C = img.shape
    device = img.device

    distance_matrix = create_distance_matrix(max_distance).to(device)
    spatial_weights = torch.exp(-distance_matrix**2 / (2 * sigma_space**2))
    loss = 0.0
    normalization = 0
    
    for dy in range(-max_distance, max_distance + 1):
        for dx in range(-max_distance, max_distance + 1):
            if (dy == 0 and dx == 0) or (dx + dy > max_distance) or (dx + dy < -max_distance):
                continue

            # Shift the image to get the neighboring pixels
            shifted_img = torch.roll(img, shifts=(-dy, -dx), dims=(0, 1))
            intensity_diff = img - shifted_img

            y_start, y_end = max(0, dy), min(H, H + dy)
            x_start, x_end = max(0, dx), min(W, W + dx)
            valid_area = [slice(y_start, y_end), slice(x_start, x_end)]
            valid_diff = intensity_diff[valid_area[0], valid_area[1], :]

            # Calculate intensity weights, and combine with spatial weights
            intensity_weights = torch.exp(-valid_diff**2 / (2 * sigma_color**2))
            combined_weights = spatial_weights[dy + max_distance, dx + max_distance] * intensity_weights
            loss += (combined_weights * valid_diff**2).mean()
            normalization += 1

    return loss / normalization


def laplacian_kernel(size, sigma):
    """ Creates a 2D Gaussian-Laplacian kernel. """
    if size % 2 == 1:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        grid = coords.unsqueeze(0).repeat(size, 1)
        kernel = torch.exp(-(grid**2 + grid.T**2) / (2 * sigma**2))
        
        center = (size - 1) // 2
        kernel[center, center] = 0
        kernel /= kernel.sum()
        kernel[center, center] = -1
    else:
        coords = torch.arange(size, dtype=torch.float32) - size // 2 +0.5
        grid = coords.unsqueeze(0).repeat(size, 1)
        kernel = torch.exp(-(grid**2 + grid.T**2) / (2 * sigma**2))
        
        center = size // 2
        kernel[center-1:center+1, center-1:center+1] = 0
        kernel /= kernel.sum()
        kernel[center-1:center+1, center-1:center+1] = -0.25
    return kernel

class Laplacian_Loss():
    def __init__(self, device, kernel_size=5, sigma=0.75, method="l1") -> None:
        self.device = device
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.method = method
        self.kernel = laplacian_kernel(kernel_size, sigma).to(self.device)
        self.kernel = self.kernel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, size, size]
        self.kernel = self.kernel.repeat(3, 1, 1, 1)  # Shape: [3, 1, size, size]
    
    def __call__(self, img):
        # Apply padding
        img_reordered = rearrange(img, 'h w c -> c h w').unsqueeze(0)
        pad_size = self.kernel_size // 2
        img_padded = F.pad(img_reordered, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

        # Convolution with the Gaussian kernel
        laplacian = F.conv2d(img_padded, self.kernel, groups=img_reordered.shape[1])
        
        if self.method == "l1":
            loss = torch.abs(laplacian).mean()
        elif self.method == "l2":
            loss = torch.pow(laplacian, 2).mean()
        return loss, laplacian.squeeze(0)
