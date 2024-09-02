import torch
import torch.nn as nn
import torch.nn.functional as nnf
from typing import Union
import numpy as np
from math import sqrt

from accelerate import load_checkpoint_in_model

import sys
sys.path.insert(0, '.')

from utils.files_utils import load_hex

from models import js_divergence


class PixelArt(nn.Module):
    def __init__(self,
                 height,
                 width,
                 palette: Union[torch.Tensor, str, np.ndarray],
                 std=1.0
                 ):
        super().__init__()
        if isinstance(palette, str):
            palette = load_hex(palette)
            palette = palette.astype(np.float32)/ 255.0
        if isinstance(palette, np.ndarray):
            palette = torch.from_numpy(palette)
        if palette.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            palette = palette.float() / 255.0

        self.register_buffer('palette', palette) # not optimized during training
        self.nb_col = palette.shape[0]
        self.std = std * sqrt(self.nb_col)
        self.img_cls = nn.Parameter(torch.randn(height, width, self.nb_col))
        self.normalize_softmax(reg = self.std)
    
    @torch.no_grad()
    def get_probability_variance(self, take_mean = True):
        img_cls_probs = nnf.softmax(self.img_cls, dim=-1)
        variance = torch.var(img_cls_probs, correction=0, dim=-1)
        if take_mean:
            return variance.mean()
        else:
            return variance
    
    @torch.no_grad()
    def get_probability_entropy(self, take_mean = True):
        img_cls_probs = nnf.softmax(self.img_cls, dim=-1)
        entropy_per_pixel = -(img_cls_probs * torch.log(img_cls_probs)).sum(dim=-1)
        if take_mean:
            return entropy_per_pixel.mean()
        else:
            return entropy_per_pixel
    
    @torch.no_grad()
    def get_probability_max(self, take_mean = True):
        img_cls_probs = nnf.softmax(self.img_cls, dim=-1)
        max_value, _ = img_cls_probs.max(dim=-1)
        if take_mean:
            return max_value.mean()
        else:
            return max_value

    def get_std(self, take_mean = True):
        if take_mean:
            return (self.img_cls.data.clone() - self.barycenter()).norm(dim=-1, keepdim=True).mean()
        else:
            return (self.img_cls.data.clone() - self.barycenter()).norm(dim=-1, keepdim=True)

    def initialize(self, target_cls):
        self.img_cls.data = target_cls.data.clone()
        self.balance()

    def barycenter(self):
        # take the average over the last dimension
        return self.img_cls.data.mean(dim=-1, keepdim=True)
    
    def balance(self):
        barycenter = self.barycenter()
        self.img_cls.data = self.img_cls.data - barycenter
        
    def normalize_softmax(self, reg = 1.0):
        self.balance()
        self.normalize_data_values(reg = reg)

    def normalize_data_values(self, reg = 1.0):
        # normalize per last channel, and multiply by reg
        self.img_cls.data = (self.img_cls.data / self.img_cls.data.norm(dim=-1, keepdim=True)) * reg
        
    def regularize(self, reg):
        self.img_cls.data = self.img_cls.data * reg

    def forward(self, smooth_softmax=False, reg = 1.0, gumbel = False, tau=1.0):
        hard = not smooth_softmax
        img_cls = self.img_cls
        if gumbel:
            img_cls_probs = nnf.gumbel_softmax(img_cls, tau=tau, hard=hard)
        else:
            img_cls_probs = nnf.softmax(img_cls*reg, dim=-1)
            if hard:
                _, max_value_indexes = img_cls_probs.max(dim=-1, keepdim=True)
                img_cls_discrete = torch.zeros_like(img_cls_probs).scatter_(-1, max_value_indexes, 1.0)
                img_cls_probs = img_cls_discrete + img_cls_probs - img_cls_probs.detach()
        img = torch.einsum('ijk,kl->ijl', img_cls_probs, self.palette)
        return img



def show_layers(checkpoint_path, palette, height, width, save):
    from matplotlib import pyplot as plt
    import os
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    if palette is None:
        palette = checkpoint['palette']
    if height is None:
        height = checkpoint['height']
    if width is None:
        width = checkpoint['width']
    print(palette)
    model = PixelArt(height, width, palette)
    palette = model.palette.clone()
    model.load_state_dict(checkpoint['renderer'])
    model.palette = palette
    img_cls_probs = nnf.softmax(model.img_cls, dim=-1) #has size HxWxN

    for i in range(model.nb_col):
        alpha = img_cls_probs[:, :, i]
        img = torch.zeros(height, width, 4)
        img[:, :, :3] = palette[i]
        img[:, :, 3] = alpha
        plt.imshow(img.detach().numpy())
        plt.axis('off')
        if save is not None:
            os.makedirs("results", exist_ok=True)
            img_np = img.detach().numpy()
            plt.imsave(f"results/layer_{i}.png", img_np)
        plt.title(f"Layer {i}")
        plt.show()
    
    img_argmax = model()
    img_softmax = model(smooth_softmax=True)
    
    plt.imshow(img_argmax.detach().numpy())
    plt.axis('off')
    if save is not None:
        os.makedirs("results", exist_ok=True)
        img_np = img_argmax.detach().numpy()
        plt.imsave(f"results/argmax.png", img_np)
    plt.title("Argmax")
    plt.show()
    
    plt.imshow(img_softmax.detach().numpy())
    plt.axis('off')
    if save is not None:
        os.makedirs("results", exist_ok=True)
        img_np = img_softmax.detach().numpy()
        plt.imsave(f"results/softmax.png", img_np)
    plt.title("Softmax")
    plt.show()
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the checkpoint to render.", required=True) # eg., 
    parser.add_argument("--palette", type=str, default=None, help="Path to the palette to render.",) # eg., assets/palettes/lospec/slowly.hex /!\ THIS PALETTE HAS TO HAVE THE SAME SIZE AS THE PALETTE USED BY THE MODEL (or left as None)
    parser.add_argument("--height", type=int, default=None, help="Height of the rendering.")
    parser.add_argument("--width", type=int, default=None, help="Width of the rendering.")
    parser.add_argument("--save_dir", type=str, default=None, help="Save the rendering.")
    args = parser.parse_args()
    
    show_layers(args.checkpoint, args.palette, args.height, args.width, args.save_dir)