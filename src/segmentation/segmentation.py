
from abc import ABC, abstractmethod
from collections import deque

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

# ####################################################################################################
#                                             Background Subtractor                                  #
# ####################################################################################################

class BackgroundSubtractor(ABC):
    @abstractmethod
    def __init__(self, frame: np.ndarray):
        self.B = np.zeros_like(frame, dtype=np.uint8)

    def _background(self) -> np.ndarray:
        return np.array(self.B, dtype=np.uint8)

    @abstractmethod
    def update(self, frame: np.ndarray | list[np.ndarray]) -> None:
        pass

    @abstractmethod
    def segment(self, frame: np.ndarray | list[np.ndarray]) -> np.ndarray:
        pass

# ####################################################################################################
#                                             Neural Network                                         #
# ####################################################################################################

# class FrameDifferencing(BackgroundSubtractor):
#     def __init__(self, frame: np.ndarray, step_size=1):
#         super().__init__(frame)
#         self.prev_frame = frame
#         self.step_size = step_size

#     def _diff_frames(self, frames):
#         # rgb version: 47 ms ± 13.7 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
#         # gray version: 441 ms ± 87.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
#         return [cv2.absdiff(frames[i], frames[i - self.step_size]) for i in range(self.step_size, len(frames), self.step_size)]

#     def _threshold_diffs(frames_diff, threshold=15, set_to=1):
#         return [cv2.threshold(frame, threshold, set_to, cv2.THRESH_BINARY)[1] for frame in [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames_diff]]

#     def _fill_masks(frames_masks):
#         return [1- binary_fill_holes(mask) for mask in frames_masks]

#     def segment(self, frame: np.ndarray, steps=1) -> np.ndarray:
#         frames_differenced = self._diff_frames(frame, step=steps)
#         frames_masks = self._threshold_diffs(frames_differenced, threshold=15)
#         frames_masks = self._fill_masks(frames_masks)

#         # frames_segmented, frames_masks = self._segment_frame_mask(
#         #     ,
#         #     frames_masks,
#         # )

#         print(f"{len(frames_segmented)} frames generated")
#         print(f"{len(frames_masks)} masks generated")

#         return frames_segmented, frames_masks

class FrameDifferencing(BackgroundSubtractor):
    def __init__(self, frame, step_size=1, threshold=15):
        self.step_size = step_size
        self.threshold = threshold
        self.buffer = deque([frame], maxlen=step_size)

    def segment(self, frame):
        diff = cv2.absdiff(frame, self.buffer[0])
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        mask = binary_fill_holes(mask > 0).astype(np.uint8) * 255
        return mask

    def update(self, frame):
        self.buffer.append(frame))

    def _background(self):
        return


# ####################################################################################################
# #                                            ViBe                                                  #
# ####################################################################################################

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