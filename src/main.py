import argparse
import logging

from .IO.image import save_output
from .IO.video import Display, load_vid
from .segmentation.segmentation import BackgroundSubtractor, FrameDifferencing, ViBe
from .utils import resize_frames

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# constants definition
DEFAULT_VIDEO_NAME = "group_1_phase1_static_indoor.mp4"
DEFAULT_SEGMENTATION_METHOD = "DIFF"
SEGMENTERS = {"DIFF": FrameDifferencing, "VIBE": ViBe}
DEFAULT_TEMPORAL_FILTERING_METHOD = "MNM"
DEFAULT_SMOOTHING_METHOD = "NONE"
DEFAULT_SIZE = (1080, 720)
DEFAULT_HEADLESS = False
SMOOTHING_METHODS = ["KERNEL", "NONE"]
DEFAULT_OUTPUT_NAME = "output.png"


def run_pipeline(
    frames,
    segmentation: str,
    size: tuple[int, int] = DEFAULT_SIZE,
    headless: bool = DEFAULT_HEADLESS,
    output_name: str = "output.png",
) -> BackgroundSubtractor:
    frames = resize_frames(frames, size=size)
    segmenter = SEGMENTERS[segmentation](frames[0])
    if headless:
        segmenter.pipeline(frames[1:])
    else:
        display = Display()
        for idx, frame in enumerate(frames):
            logger.info(f"Frame {idx + 1}/{len(frames)}")
            segmenter.segment(frame)
            segmenter.update(frame)
            display.update(frame, segmenter._background())

    # save_output(segmenter._background(), output_name)
    save_output(segmenter.ghosts(), output_name)

    return segmenter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_NAME)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--step-frames", type=int, default=10)
    parser.add_argument("--segmentation", type=str, default=DEFAULT_SEGMENTATION_METHOD)
    parser.add_argument(
        "--filtering", type=str, default=DEFAULT_TEMPORAL_FILTERING_METHOD
    )
    parser.add_argument(
        "--smoothing",
        type=str,
        default=DEFAULT_SMOOTHING_METHOD,
        choices=SMOOTHING_METHODS,
    )
    parser.add_argument("--size", type=int, nargs=2, default=DEFAULT_SIZE)
    parser.add_argument("--headless", type=bool, default=DEFAULT_HEADLESS)
    parser.add_argument("--output", type=str, default="output.png")
    args = parser.parse_args()

    frames = load_vid(
        args.video, selected_frames=range(0, args.max_frames, args.step_frames)
    )
    logger.info(f"Number of frames loaded: {len(frames)}")

    run_pipeline(
        frames, args.segmentation, tuple(args.size), args.headless, args.output
    )
