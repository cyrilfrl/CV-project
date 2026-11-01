from abc import ABC, abstractmethod
from collections import deque

import cv2
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

    def pipeline(self, frames: np.ndarray | list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        for frame in frames:
            self.segment(frame)
            self.update(frame)

# ####################################################################################################
#                                            Frame Differencing                                      #
# ####################################################################################################

class FrameDifferencing(BackgroundSubtractor):
    def __init__(self, frame, step_size=30, threshold=15, alpha=0.05):
        super().__init__(frame)
        self.step_size = step_size
        self.threshold = threshold
        self.alpha = alpha
        self.buffer = deque([frame], maxlen=step_size)
        self.history = []

    def update(self, frame: np.ndarray) -> None:
        self.buffer.append(frame)
        candidate_B = self.alpha * frame + (1 - self.alpha) * self.B
        keep = self._mask == 0
        self.B[keep] = candidate_B[keep]
        self.history.append((frame.copy(), self._mask.copy()))
        

    def segment(self, frame):
        diff = cv2.absdiff(frame, self.buffer[0])
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        mask = binary_fill_holes(mask > 0).astype(np.uint8) * 255
        self._mask = mask
        return mask

    def ghosts(self, k: int = 15) -> np.array:
        ghost_img = self.B.copy().astype(np.float32)
        N = len(self.history)

        index = np.linspace(N - 1 - 5 * k, N - 1, k, dtype=int)
        opacity = np.linspace(1.0 / k, 1.0, k)
        
        for i, idx in enumerate(index):
            f, m = self.history[idx]
            fg_mask = m > 0
            ghost_img[fg_mask] = (opacity[i] * f[fg_mask] + (1 - opacity[i]) * ghost_img[fg_mask])
        return ghost_img.astype(np.uint8) 

# ####################################################################################################
# #                                            ViBe                                                  #
# ####################################################################################################

class ViBe(BackgroundSubtractor):
    def __init__(self, frame, N=20, R=20, min_count=2, time_sampling_prob=0.15):
        super().__init__(frame)
        self.models = None
        self.N, self.R = N, R # N: number of samples per pixel, R: radius for matching [RGB distance]
        self.min_count = min_count # minimum number of matches to consider a pixel as background
        self.time_sampling_prob = time_sampling_prob # probability of updating the model at each time step
        self.samples = self._initialize(frame) # (h, w, N, 3)
        self.h, self.w, _ = frame.shape
        self.rows, self.cols = np.mgrid[:self.h, :self.w]   # rows[i, j] = i, cols[i, j] = j
        self.history = []

    def _initialize(self, frame):
        h, w, _ = frame.shape
        padded = np.pad(frame, ((1, 1), (1, 1), (0, 0)), mode="reflect")
        yy, xx = np.mgrid[:h, :w]
        LOWER, UPPER = -1, 1
        dy = np.random.randint(LOWER, UPPER + 1, (h, w, self.N))
        dx = np.random.randint(LOWER, UPPER + 1, (h, w, self.N))
        rows_3d = yy[:, :, None] # broadcast to shape (h, w, 1)
        cols_3d = xx[:, :, None] # broadcast to shape (h, w, 1)
        neighbour_rows = rows_3d + 1 + dy # (h, w, N) ! broadcast to shape (h, w, N)
        neighbour_cols = cols_3d + 1 + dx # +1 for padding offset
        samples = padded[neighbour_rows, neighbour_cols]
        return samples # (h, w, N, 3)

    def segment(self, frame):
        diff = self.samples.astype(np.int32) - frame[:, :, None, :] # (h, w, N, 3)
        counts = np.sum((np.sum(diff**2, axis=-1) < self.R**2), axis=-1) # (h, w)
        self._mask = np.where(counts < self.min_count, 255, 0).astype(np.uint8)
        return self._mask

    def update(self, frame):
        # select if nearby current model pixels and if random number < time_sampling_prob
        selected = (self._mask == 0) & (np.random.rand(self.h, self.w) < self.time_sampling_prob)  # (h, w) bool
        values = frame[selected]  # (nb_selected, 3), E[nb_selected] = time_sampling_prob*(h*w)
        nb_selected = values.shape[0]

        ys = self.rows[selected] # (nb_selected,)
        xs = self.cols[selected] # (nb_selected,)

        # which neighboring pixel to update
        LOWER, UPPER = -1, 1
        ny = np.clip(ys + np.random.randint(LOWER, UPPER + 1, nb_selected), 0, self.h - 1) # clip indexing to boundaries
        nx = np.clip(xs + np.random.randint(LOWER, UPPER + 1, nb_selected), 0, self.w - 1)
        nth_plane = np.random.randint(0, self.N, nb_selected) # plane to update
        self.samples[ny, nx, nth_plane] = values

        self.B = np.median(self.samples, axis=2).astype(np.float32)

        self.history.append((frame.copy(), self._mask.copy()))

    def ghosts(self, k: int = 15) -> np.ndarray:
        ghost_img = self.B.copy().astype(np.float32)
        N = len(self.history)

        index = np.linspace( N - 1 - 5 * k, N - 1, k, dtype=int)
        opacity = np.linspace(1.0 / k, 1.0, k)
        
        for i, idx in enumerate(index):
            f, m = self.history[idx]
            fg_mask = m > 0
            
            ghost_img[fg_mask] = (opacity[i] * f[fg_mask] + (1 - opacity[i]) * ghost_img[fg_mask])
            
        return ghost_img.astype(np.uint8)