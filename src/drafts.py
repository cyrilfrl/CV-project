
# VIBE

    # def _initialize(self, frame: np.ndarray):
    #     h, w, _ = frame.shape
    #     padded = np.pad(frame, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    #     offsets = np.array(
    #         [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    #     )
    #     choices = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=(h, w, self.N))
    #     choices_offsets = offsets[choices]  # (h, w, 20, 2)
    #     choices_offsets_x = choices_offsets[:, :, :, 0]
    #     choices_offsets_y = choices_offsets[:, :, :, 1]

    #     xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    #     coors_x = xx[:, :, None] + choices_offsets_x  # (h, w, 20)
    #     coors_y = yy[:, :, None] + choices_offsets_y  # (h, w, 20)

    #     self.models = np.zeros((h, w, 3, self.N), dtype=np.uint8)  # (h, w, 3, 20)
    #     self.models[:, :, :, :] = padded[coors_y, coors_x, :].transpose(
    #         0, 1, 3, 2
    #     )  # note: coors_x and _y broadcasting to (h, w, 20) inserted before channel

    
    # def _count_at_distance(self, frame):
    #     diff = self.models - frame[:, :, :, None]
    #     sq_dist = np.sum(diff**2, axis=2)
    #     dist = np.sqrt(sq_dist)
    #     close = dist < self.R
    #     counts = np.sum(close, axis=2)
    #     return counts


    # def update(self, frame: np.ndarray):
    #     if self.models is None:
    #         self.models = self._initialize(frame)
    #     elif isinstance(self.models, np.ndarray):
    #         counts = self._count_at_distance(frame)
    #         to_update = self._to_update(frame)

    #         h, w, _ = frame.shape
    #         for i in range(h):
    #             for j in range(w):
    #                 if to_update[i, j]:
    #                     if counts[i, j] < self.min_count:
    #                         continue

    #                     n = np.random.randint(0, self.N)
    #                     self.models[i, j, :, n] = frame[i, j, :]

    #                     n = np.random.randint(0, self.N)
    #                     i_neighbor, j_neighbor = self._get_random_neighboring_pixel(
    #                         j, i, frame
    #                     )
    #                     self.models[i_neighbor, j_neighbor, :, n] = frame[i, j, :]

    # def update_and_display(self, frame: np.ndarray, display_size=(1920, 1080)):
    #     self.update(frame)
    #     current_model = np.median(self.models, axis=3).astype(np.uint8)

    #     # half display size
    #     display_size = (display_size[0] // 2, display_size[1] // 2)

    #     model_resized = cv2.resize(current_model, display_size)
    #     frame_resized = cv2.resize(frame, display_size)
    #     screen = cv2.hconcat([model_resized, frame_resized])
    #     cv2.imshow("ViBe", screen)
    #     cv2.waitKey(5)

    # def _to_update(self, frame):
    #     h, w, _ = frame.shape
    #     return np.random.rand(h, w) <= self.time_sampling_prob

    # def _get_random_neighboring_pixel(self, coord_x, coord_y, frame):
    #     h, w, _ = frame.shape

    #     row = np.random.randint(max(0, coord_y - 1), min(coord_y + 2, h - 1))
    #     if row == coord_y - 1 or row == coord_y + 1:
    #         col = np.random.randint(max(0, coord_x - 1), min(coord_x + 2, w - 1))
    #     elif row == coord_y:
    #         col = np.random.choice([max(0, coord_x - 1), min(coord_x + 1, w - 1)])

    #     return (row, col)