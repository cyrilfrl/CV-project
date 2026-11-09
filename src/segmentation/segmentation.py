from matplotlib.pylab import var
from abc import ABC, abstractmethod
from collections import deque

import cv2
from matplotlib.pyplot import hist
import numpy as np
from scipy.ndimage import binary_fill_holes

# FILTERS = {"MNM": mean_non_masked}
# SMOOTHERS = {
#     "KERNEL": lambda img: smooth_image(img, kernel_size=5),
#     "NONE": lambda img: img,
# }

# ####################################################################################################
#                                             Background Subtractor                                  #
# ####################################################################################################


class BackgroundSubtractor(ABC):
    @abstractmethod
    def __init__(self, frame: np.ndarray):
        self.B = frame.astype(np.float32)
        self._mask = np.zeros_like(frame, dtype=np.uint8)

    def _background(self) -> np.ndarray:
        return self.B.astype(np.uint8)

    @abstractmethod
    def update(self, frame: np.ndarray) -> None:
        pass

    @abstractmethod
    def segment(self, frame: np.ndarray) -> np.ndarray:
        pass

    def pipeline(self, frames: np.ndarray | list[np.ndarray]) -> None:
        for frame in frames:
            self.segment(frame)
            self.update(frame)


# ####################################################################################################
#                                            Frame Differencing                                      #
# ####################################################################################################

class FrameDifferencing(BackgroundSubtractor):
    def __init__(
        self, frame, step_size=30, threshold=15, alpha=0.05, ghost_past: int = 150
    ):
        super().__init__(frame)
        self.step_size = step_size
        self.threshold = threshold
        self.alpha = alpha
        self.ghost_past = ghost_past
        self.buffer = deque([(frame, self._mask)], maxlen=ghost_past)

    def segment(self, frame):
        diff_frame = (
            self.buffer[0][0]
            if len(self.buffer) < self.step_size
            else self.buffer[-self.step_size][0]
        )

        # does not know wich one is the original (symetric)
        diff = cv2.cvtColor(cv2.absdiff(frame, diff_frame), cv2.COLOR_BGR2GRAY) > self.threshold

        # so compare also with background
        diff_bg = cv2.cvtColor(cv2.absdiff(frame, self.B.astype(np.uint8)), cv2.COLOR_BGR2GRAY) > self.threshold

        mask = (diff & diff_bg).astype(np.uint8) * 255
        self._mask = mask
        return mask


    def update(self, frame: np.ndarray) -> None:
        self.buffer.append((frame, self._mask))
        candidate_B = self.alpha * frame + (1 - self.alpha) * self.B
        keep = self._mask == 0
        self.B[keep] = candidate_B[keep]


    def ghosts(self, ghost_number: int = 15) -> np.ndarray:
        ghost_img = self.B.copy().astype(np.float32)

        index = np.linspace(-self.ghost_past, -1, ghost_number, dtype=int)
        opacity = np.linspace(1.0 / ghost_number, 1.0, ghost_number)

        for i, idx in enumerate(index):
            f, m = self.buffer[idx]
            fg_mask = m > 0
            ghost_img[fg_mask] = (
                opacity[i] * f[fg_mask] + (1 - opacity[i]) * ghost_img[fg_mask]
            )

        return ghost_img.astype(np.uint8)


# ####################################################################################################
# #                                            ViBe                                                  #
# ####################################################################################################


class ViBe(BackgroundSubtractor):
    def __init__(self, frame, N=20, R=20, min_count=2, time_sampling_prob=0.15, ghost_past=150):
        super().__init__(frame)
        self.models = None
        self.h, self.w, _ = frame.shape
        self.N, self.R =  N, R  # N: number of samples per pixel, R: radius for matching [RGB distance]
        self.min_count = min_count # minimum number of matches to consider a pixel as background
        self.time_sampling_prob = time_sampling_prob # probability of updating the model at each time step
        self.shape_buffer = (self.h, self.w, self.N, 3)    # include padding in the size
        self.samples = self._initialize(frame)  # (h, w, N, 3)
        self.rows, self.cols = np.mgrid[
            : self.h, : self.w
        ]  # rows[i, j] = i, cols[i, j] = j
        self.ghost_past = ghost_past
        self.buffer = deque([(frame, self._mask)], maxlen=ghost_past)

    def _initialize(self, frame):
        frame_buffer = frame[:, :, None, :]
        LOWER, UPPER = -5, 5
        noise = np.random.randint(LOWER, UPPER + 1, self.shape_buffer)
        samples = np.clip(frame_buffer + noise, 0, 255).astype(np.uint8)
        return samples  # (h, w, N, 3)

    def segment(self, frame):
        diff = self.samples.astype(np.int32) - frame[:, :, None, :]  # (h, w, N, 3)
        counts = np.sum((np.sum(diff**2, axis=-1) < self.R**2), axis=-1)  # (h, w)
        self._mask = np.where(counts < self.min_count, 255, 0).astype(np.uint8)
        return self._mask

    def update(self, frame):
        # select if nearby current model pixels and if random number < time_sampling_prob
        selected = (self._mask == 0) & (
            np.random.rand(self.h, self.w) < self.time_sampling_prob
        )  # (h, w) bool
        values = frame[ selected ]  # (nb_selected, 3), E[nb_selected] = time_sampling_prob*(h*w)
        nb_selected = values.shape[0]

        ys = self.rows[selected]  # (nb_selected,)
        xs = self.cols[selected]  # (nb_selected,)

        # which neighboring pixel to update
        LOWER, UPPER = -1, 1
        ny = np.clip(
            ys + np.random.randint(LOWER, UPPER + 1, nb_selected), 0, self.h - 1
        )  # clip indexing to boundaries
        nx = np.clip(
            xs + np.random.randint(LOWER, UPPER + 1, nb_selected), 0, self.w - 1
        )
        nth_plane = np.random.randint(0, self.N, nb_selected)  # plane to update
        self.samples[ny, nx, nth_plane] = values

        self.B = np.median(self.samples, axis=2).astype(np.float32)

        self.buffer.append((frame, self._mask))

    def ghosts(self, ghost_number: int = 15) -> np.ndarray:
        ghost_img = self.B.copy().astype(np.float32)

        index = np.linspace(-self.ghost_past, -1, ghost_number, dtype=int)
        opacity = np.linspace(1.0 / ghost_number, 1.0, ghost_number)

        for i, idx in enumerate(index):
            f, m = self.buffer[idx]
            fg_mask = m > 0

            ghost_img[fg_mask] = (
                opacity[i] * f[fg_mask] + (1 - opacity[i]) * ghost_img[fg_mask]
            )

        return ghost_img.astype(np.uint8)


# ####################################################################################################
#                                            Mixture of Gaussian                                     #
# ####################################################################################################

class MixtureOfGaussian(BackgroundSubtractor):
    def __init__(
        self, frame, ghost_past: int = 150
    ):
        super().__init__(frame)
        self.backsub = cv2.createBackgroundSubtractorMOG2(history=200)
        self.ghost_past = ghost_past
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_OPEN, (5, 5))
        self.kernel_close= cv2.getStructuringElement(cv2.MORPH_CLOSE, (30, 30))
        self.buffer = deque([(frame, self._mask)], maxlen=ghost_past)

    def segment(self, frame):
        self._mask = self.backsub.apply(frame)
        _, self._mask = cv2.threshold(self._mask, 127, 255, cv2.THRESH_BINARY)      # set shadows to background
        self._mask = cv2.morphologyEx(self._mask, cv2.MORPH_OPEN, self.kernel_open)      # OPEN to remove noise
        self._mask = cv2.morphologyEx(self._mask, cv2.MORPH_CLOSE, self.kernel_close)     # CLOSE to fill holes
        self.buffer.append((frame, self._mask))
        return self._mask

    def update(self, frame: np.ndarray) -> None:
        self.B = self.backsub.getBackgroundImage()
        self.B = cv2.morphologyEx(self.B, cv2.MORPH_OPEN, self.kernel_open)


    def ghosts(self, ghost_number: int = 15) -> np.ndarray:
        ghost_img = self.B.copy().astype(np.float32)

        index = np.linspace(-self.ghost_past, -1, ghost_number, dtype=int)
        opacity = np.linspace(1.0 / ghost_number, 1.0, ghost_number)

        for i, idx in enumerate(index):
            f, m = self.buffer[idx]
            fg_mask = m > 0
            ghost_img[fg_mask] = (
                opacity[i] * f[fg_mask] + (1 - opacity[i]) * ghost_img[fg_mask]
            )

        return ghost_img.astype(np.uint8)
