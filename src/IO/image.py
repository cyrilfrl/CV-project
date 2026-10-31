
import logging
from pathlib import Path

import cv2
import numpy as np

from ..utils import _get_parent_dir

logger = logging.getLogger(__name__)


def _get_output_path(video_name: str) -> str:
    try:
        parent_dir = _get_parent_dir()
    except FileNotFoundError as e:
        logger.error(e)
        raise
    path = parent_dir / "outputs" / Path(video_name)
    return str(path)

def save_output(frame: np.ndarray, output_name: str) -> None:
    output_path = _get_output_path(output_name)
    cv2.imwrite(output_path, frame)
    logger.info(f"Output saved to: {output_path}")