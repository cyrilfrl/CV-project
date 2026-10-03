
import argparse
import logging

import cv2
import numpy as np

from .filters import mean_non_masked, smooth_image
from .IO.video import load_vid
from .segmentation import segment_frame_differencing, segment_people_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# constants definition
DEFAULT_VIDEO_NAME="People Walking Free Stock Footage, Royalty-Free No Copyright Content.mp4"
DEFAULT_SEGMENTATION_METHOD="DNN"
DEFAULT_TEMPORAL_FILTERING_METHOD="MNM"
DEFAULT_SMOOTHING_METHOD="NONE"

SEGMENTERS = {"DNN": segment_people_batch, "DIFF": segment_frame_differencing}
FILTERS = {"MNM": mean_non_masked}
SMOOTHERS = {
    "KERNEL": lambda img: smooth_image(img, kernel_size=5),
    "NONE": lambda img: img,
}

def run_pipeline(frames, segmentation: str, filtering: str, smoothing: str) -> np.ndarray:
    frames_masked, frames_masks = SEGMENTERS[segmentation](frames)
    img = FILTERS[filtering](frames_masked, frames_masks)
    img = SMOOTHERS[smoothing](img)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_NAME)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--step-frames", type=int, default=10)
    parser.add_argument("--segmentation", type=str, default=DEFAULT_SEGMENTATION_METHOD)
    parser.add_argument("--filtering", type=str, default=DEFAULT_TEMPORAL_FILTERING_METHOD)
    parser.add_argument("--smoothing", type=str, default=DEFAULT_SMOOTHING_METHOD, choices=["KERNEL", "NONE"])
    args = parser.parse_args()

    frames = load_vid(args.video, selected_frames=range(0, args.max_frames, args.step_frames))
    logger.info(f"Number of frames loaded: {len(frames)}")

    # REMOVE THIS AS SOON AS POSSIBLE
    frames = [cv2.resize(frame, (960, 540)) for frame in frames]
    from segmentation import ViBe
    model = ViBe()

    idx = 0
    for idx, frame in enumerate(frames):
        model.update_and_display(frame)
        print(f"Processed frame {idx + 1}/{len(frames)}")

    cv2.destroyAllWindows()


    # img = run_pipeline(frames, args.segmentation, args.filtering, args.smoothing)

    # # save image
    # save_output(f"output_{args.segmentation}_{args.filtering}_{args.smoothing}_{args.step_frames}.png", img)