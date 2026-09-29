
import cv2
import numpy as np


# temporal filters
def _robust_mean(sum, count):
    count[count == 0] = -1
    mean = sum / np.stack([count]*3, axis=-1)
    mean = mean.astype(np.uint8)
    mean[mean < 0] = 0
    return mean

# def mean_non_masked(frames_masked, frames_masks):
#     sum = np.sum(frames_masked, axis=0)
#     count = frames_masks.sum(axis=0)
#     mean = _robust_mean(sum, count)
#     return mean

# # spatial filters
# def smooth_image(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
#     kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
#     img_smoothed = cv2.filter2D(img, ddepth=cv2.CV_8U, kernel=kernel)
#     return img_smoothed




def mean_non_masked(
    frames_masked: np.ndarray,
    frames_masks: np.ndarray,
) -> np.ndarray:
    frames_array = np.asarray(frames_masked, dtype=np.float32)
    masks_array = np.asarray(frames_masks)

    if frames_array.ndim != 4:
        raise ValueError(
            f"Expected frames with shape (T, H, W, C), got {frames_array.shape}"
        )

    if masks_array.shape != frames_array.shape[:3]:
        raise ValueError(
            "Mask shape must be (T, H, W), "
            f"got {masks_array.shape} for frames {frames_array.shape}"
        )

    # Every nonzero mask value means the pixel is valid/background.
    valid = (masks_array != 0)[..., None]

    pixel_sum = np.sum(
        np.where(valid, frames_array, 0.0),
        axis=0,
    )

    pixel_count = np.sum(valid, axis=0)

    mean = np.divide(
        pixel_sum,
        pixel_count,
        out=np.zeros_like(pixel_sum),
        where=pixel_count != 0,
    )

    return np.clip(mean, 0, 255).astype(np.uint8)


def smooth_image(
    image: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.float32,
    ) / (kernel_size * kernel_size)

    return cv2.filter2D(
        image,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT,
    )