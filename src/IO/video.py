import logging
from pathlib import Path

import cv2
import numpy as np

from ..utils import _get_parent_dir

logger = logging.getLogger(__name__)

def _get_video_path(video_name: str) -> str:
    try:
        parent_dir = _get_parent_dir()
    except FileNotFoundError as e:
        logger.error(e)
        raise
    path = parent_dir / "data" / Path(video_name)
    return str(path)

def load_vid(video_name: str, selected_frames: set[int] | None = None) -> list[np.ndarray]:
    capture = cv2.VideoCapture(_get_video_path(video_name))
    fps = capture.get(cv2.CAP_PROP_FPS)
    logger.info(f"FPS: {fps}")
    logger.debug(f"Path: {_get_video_path(video_name)}")

    records = {}
    idx = 0

    while True:
        ret, frame = capture.read()
        if not ret or max(selected_frames) < idx:
            break
        records[idx] = {"frame_idx": idx, "frame": frame}
        idx += 1

    capture.release()
    return [records[idx]["frame"] for idx in records if selected_frames is None or idx in selected_frames]