
import cv2
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes
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


class FrameDifferencing(Segmenter):
    def diff_frames(frames, step=20):
        # rgb version: 47 ms ± 13.7 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
        # gray version: 441 ms ± 87.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        return [cv2.absdiff(frames[i], frames[i - step]) for i in range(step, len(frames), step)]

    def threshold_diffs(frames_diff, threshold=15, set_to=1):
        return [cv2.threshold(frame, threshold, set_to, cv2.THRESH_BINARY)[1] for frame in [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames_diff]]

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

####################################################################################################
#                                            ViBe                                                  #
####################################################################################################

class ViBe:
    def __init__(self, N=20, R=20, min_count=2, time_sampling_prob=0.0625):
        self.models = None
        self.N = N
        self.R = R
        self.min_count = min_count
        self.time_sampling_prob = time_sampling_prob

    def _initialize(self, frame: np.ndarray):
        h, w, _ = frame.shape
        padded = np.pad(frame, ((1, 1), (1, 1), (0, 0)), mode="reflect")
        offsets = np.array([ (-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1) ])
        choices = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=(h, w, self.N))
        choices_offsets = offsets[choices] # (h, w, 20, 2)
        choices_offsets_x = choices_offsets[:, :, :, 0]
        choices_offsets_y = choices_offsets[:, :, :, 1]

        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        coors_x = xx[:, :, None] + choices_offsets_x # (h, w, 20)
        coors_y = yy[:, :, None] + choices_offsets_y # (h, w, 20)

        self.models = np.zeros((h, w, 3, self.N), dtype=np.uint8) # (h, w, 3, 20)
        self.models[:, :, :, :] = padded[coors_y, coors_x, :].transpose(0, 1, 3, 2) # note: coors_x and _y broadcasting to (h, w, 20) inserted before channel

    def _count_at_distance(self, frame):
        diff = self.models - frame[:, :, :, None]
        sq_dist = np.sum(diff**2, axis=2)
        dist = np.sqrt(sq_dist)
        close = dist < self.R
        counts = np.sum(close, axis=2)
        return counts

    def _to_update(self, frame):
        h, w, _ = frame.shape
        return np.random.rand(h, w) <= self.time_sampling_prob

    def _get_random_neighboring_pixel(self, coord_x, coord_y, frame):
        h, w, _ = frame.shape
        
        row = np.random.randint(max(0, coord_y - 1), min(coord_y + 2, h - 1))
        if row == coord_y - 1 or row == coord_y + 1:
            col = np.random.randint(max(0, coord_x - 1), min(coord_x + 2, w - 1))
        elif row == coord_y:
            col = np.random.choice([max(0, coord_x - 1), min(coord_x + 1, w - 1)])

        return (row, col)

    def update(self, frame: np.ndarray):
        if self.models is None:
            self._initialize(frame)
        elif isinstance(self.models, np.ndarray):
            counts = self._count_at_distance(frame)
            to_update = self._to_update(frame)

            h, w, _ = frame.shape
            for i in range(h):
                for j in range(w):
                    if to_update[i, j]:
                        if counts[i, j] < self.min_count:
                            continue

                        n = np.random.randint(0, self.N)
                        self.models[i, j, :, n] = frame[i, j, :]

                        n = np.random.randint(0, self.N)
                        i_neighbor, j_neighbor = self._get_random_neighboring_pixel(j, i, frame)
                        self.models[i_neighbor, j_neighbor, :, n] = frame[i, j, :]
    
    def update_and_display(self, frame: np.ndarray, display_size=(1920, 1080)):
        self.update(frame)
        current_model = np.median(self.models, axis=3).astype(np.uint8)

        # half display size
        display_size = (display_size[0] // 2, display_size[1] // 2)

        model_resized = cv2.resize(current_model, display_size)
        frame_resized = cv2.resize(frame, display_size)
        screen = cv2.hconcat([model_resized, frame_resized])
        cv2.imshow("ViBe", screen)
        cv2.waitKey(5)

    def segmentation(self, frame: np.ndarray):
        pass