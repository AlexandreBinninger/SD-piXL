from PIL import Image, ImageFilter
import numpy as np
import cv2

def canny_edge(image: Image, min_threshold = 100, max_threshold = 200, blur = False, blur_radius = 1):
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    img = np.array(image)
    img = cv2.Canny(img, min_threshold, max_threshold)
    img = img[:, :, None]
    img = np.concatenate([img, img, img], axis=2)
    img = Image.fromarray(img)
    return img
