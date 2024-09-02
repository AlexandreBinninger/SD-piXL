from .downsample import nearest_downsample, kmeans_downsample, palette_downsample
from .loss_regularizer import total_variation_loss, bilateral_edge_preserving_loss, Laplacian_Loss, FFT_Loss, gradient_loss
from .depth_estimator import DepthEstimator
from .canny_edge import canny_edge