
import cv2
import numpy as np
import torch
from torchvision.models.segmentation import (
    LRASPP_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,
)

PERSON_CLASS_ID = 15  # "person" index in this model's label set

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    model = lraspp_mobilenet_v3_large(weights=weights)
    model.eval()
    return model.to(device)

model = load_model()

def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    return tensor.to(device)


def segment_people_batch(frames, model=model, batch_size=8, dilate_kernel_size=15) -> np.ndarray:
    """Return frames with people zeroed out, and a (T, H, W) mask where
    1 = background (valid for reconstruction), 0 = person (excluded)."""
    person_masks = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))

    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        tensors = torch.stack([preprocess(f) for f in batch])

        with torch.inference_mode():
            output = model(tensors)["out"]

        class_maps = output.argmax(dim=1)
        batch_masks = (class_maps == PERSON_CLASS_ID).cpu().numpy().astype(np.uint8) * 255

        for mask in batch_masks:
            dilated = cv2.dilate(mask, kernel, iterations=1)
            person_masks.append(dilated)

    frames_masked = []
    background_masks = []
    for frame, person_mask in zip(frames, person_masks):
        background_mask = (person_mask == 0).astype(np.uint8)  # 1 = background, 0 = person
        frame_masked = frame * background_mask[..., None]      # zero out person pixels
        frames_masked.append(frame_masked)
        background_masks.append(background_mask.astype(np.int8))

    return np.asarray(frames_masked), np.asarray(background_masks)

from abc import ABC


class Segmenter(ABC):
    """
    Segment moving objects from a set of consecutive frames or out of sample frames.
    """
    @staticmethod
    def segment_frame_single(img: np.ndarray):
        pass

    @staticmethod
    def segment_frames_batch(imgs: list[np.ndarray]):
        pass

class GRABCUT:
    pass





from scipy.ndimage import binary_fill_holes


def diff_frames(frames, step=20):
    # stay rgb version: 47 ms ± 13.7 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    # convert gray version: 441 ms ± 87.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    return [cv2.absdiff(frames[i], frames[i - step]) for i in range(step, len(frames), step)]

def threshold_diffs(frames_diff, threshold=15):
    # return [np.where(frame > threshold, 1, 0).astype(np.uint8) for frame in frames_diff]
    return [cv2.threshold(frame, threshold, 1, cv2.THRESH_BINARY)[1] for frame in [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames_diff]]

def fill_masks(frames_masks):
    return [1- binary_fill_holes(mask) for mask in frames_masks]

def segment_frame_mask(imgs, masks): # AI BULLSHIT
    if not masks:
        raise ValueError("No masks were generated")

    n_imgs = len(imgs)
    n_masks = len(masks)
    chunk_size = max(1, n_imgs // n_masks)

    masked_frames = []
    expanded_masks = []

    for index, image in enumerate(imgs):
        mask_index = min(index // chunk_size, n_masks - 1)
        mask = np.asarray(masks[mask_index], dtype=np.uint8)

        masked_frames.append(image * mask[..., None])
        expanded_masks.append(mask)

    return (
        np.asarray(masked_frames),
        np.asarray(expanded_masks),
    )

# def segment_frame_differencing(imgs):
#     frames_differenced = diff_frames(imgs, step=20)
#     frames_masks = threshold_diffs(frames_differenced, threshold=15)
#     frames_masks = fill_masks(frames_masks)
#     frames_segmented = segment_frame_mask(imgs, frames_masks)

#     print(len(imgs), "masks generated")
#     print(len(frames_masks), "masks generated")

#     return frames_segmented, frames_masks

def segment_frame_differencing(imgs):
    frames_differenced = diff_frames(imgs, step=1)
    frames_masks = threshold_diffs(frames_differenced, threshold=15)
    frames_masks = fill_masks(frames_masks)

    frames_segmented, frames_masks = segment_frame_mask(
        imgs,
        frames_masks,
    )

    print(f"{len(frames_segmented)} frames generated")
    print(f"{len(frames_masks)} masks generated")

    return frames_segmented, frames_masks