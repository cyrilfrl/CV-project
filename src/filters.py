
import cv2
import numpy as np


# temporal filters
def _robust_mean(sum, count):
    count[count == 0] = -1
    mean = sum / np.stack([count]*3, axis=-1)
    mean = mean.astype(np.uint8)
    mean[mean < 0] = 0
    return mean

def mean_non_masked(frames_masked, frames_masks):
    sum = np.sum(frames_masked, axis=0)
    count = frames_masks.sum(axis=0)
    mean = _robust_mean(sum, count)
    return mean

# spatial filters
def smooth_image(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    img_smoothed = cv2.filter2D(img, ddepth=cv2.CV_8U, kernel=kernel)
    return img_smoothed