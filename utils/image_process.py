from einops import rearrange
import numpy as np

def get_categories(img : np.ndarray):
    img = rearrange(img, "h w c -> (h w) c")
    categories = np.unique(img, axis=0)
    categories_sorted_index = np.lexsort(categories.T[::-1])
    categories = categories[categories_sorted_index]
    return categories

def set_categories_img(img: np.ndarray, cat: np.ndarray):
    height, width, channels = img.shape
    img = rearrange(img, "h w c -> (h w) c")
    distances = np.sum(np.abs(img[:, np.newaxis, :] - cat[np.newaxis, :, :]), axis = -1)
    img_cat = np.argmin(distances, axis=1)
    img_cat = rearrange(img_cat, "(h w) -> h w", h=height, w=width)
    return img_cat
