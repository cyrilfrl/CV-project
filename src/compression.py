
import logging

import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compress(img, target_size=(720, 1280)):
    h_target, w_target = target_size
    h_current, w_current, _ = img.shape
    logger.info(f"Original dimensions: {w_current}x{h_current}, target dimensions: {w_target}x{h_target}")

    return cv2.resize(img, (w_target, h_target))

def uncompress(img, target_size=(720, 1280)):
    h_target, w_target = target_size
    h_current, w_current, _ = img.shape
    logger.info(f"Original dimensions: {w_current}x{h_current}, target dimensions: {w_target}x{h_target}")

    return cv2.resize(img, (w_target, h_target))
