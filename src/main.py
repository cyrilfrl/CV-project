import argparse
import logging

from .IO.image import save_output
from .IO.video import Display, load_vid
from .segmentation.segmentation import BackgroundSubtractor, FrameDifferencing, ViBe, MixtureOfGaussian
from .utils import resize_frames
from .stitching.stitching import stitching_built_in, stitch, get_homeography

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# constants definition
DEFAULT_VIDEO_NAME = "group_1_phase3_pan_dynamic_indoor.mp4"
DEFAULT_SEGMENTATION_METHOD = "DIFF"
SEGMENTERS = {"DIFF": FrameDifferencing, "VIBE": ViBe, "MOG2": MixtureOfGaussian}
DEFAULT_TEMPORAL_FILTERING_METHOD = "MNM"
DEFAULT_SMOOTHING_METHOD = "NONE"
DEFAULT_SIZE = (1080, 720)
DEFAULT_HEADLESS = False
SMOOTHING_METHODS = ["KERNEL", "NONE"]
DEFAULT_OUTPUT_NAME = "output.png"


def phase1(frames, segmentation: str, headless: bool = DEFAULT_HEADLESS):
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
    
    return segmenter._background(), segmenter.ghosts()


import cv2
import numpy as np

def phase3(frames, segmentation: str, headless: bool = DEFAULT_HEADLESS):
    display = None if headless else Display()

    ## Initialization
    h_img, w_img = frames[0].shape[:2]
    # Panorama size
    pano_w, pano_h = w_img * 4, int(h_img * 1.5)

    # Placement of the 1st frame on the left of the panorama
    T_offset = np.array([
        [1, 0, w_img // 2],
        [0, 1,  h_img // 4],
        [0, 0, 1]
    ], dtype=np.float32)

    # Set 1st frame on the panorama
    first_warped = cv2.warpPerspective(frames[0], T_offset, (pano_w, pano_h))
    seen_mask = cv2.warpPerspective(
        np.ones((h_img, w_img), dtype=np.uint8) * 255, T_offset, (pano_w, pano_h)
    )

    # 1st Homoegraphy matrix
    H_t_to_0 = np.eye(3, dtype=np.float32)

    # Avoid black line due to interpolation when blending
    inner_mask = cv2.erode(
        np.ones((h_img, w_img), dtype=np.uint8) * 255,
        np.ones((5, 5), np.uint8)
    )

    # Segmenter
    segmenter = SEGMENTERS[segmentation](first_warped) 

    # Background
    clean_panorama = first_warped.copy()
    
    for idx, frame in enumerate(frames):
        logger.info(f"Frame {idx + 1}/{len(frames)}")

        # Get the homeography matrix
        if idx > 0:
            H_step = get_homeography(frame, frames[idx - 1])
            H_t_to_0 = H_t_to_0 @ H_step
        H_to_pano = T_offset @ H_t_to_0
        # Warp the current frame
        warped_frame = cv2.warpPerspective(frame, H_to_pano, (pano_w, pano_h))
        
        # Zone where we can blend the frame safely, flags=cv2.INTER_NEAREST remove some black lines
        valid_zone = cv2.warpPerspective(inner_mask, H_to_pano, (pano_w, pano_h), flags=cv2.INTER_NEAREST)

        # Add part of the frame where it has never been seen (where the panorama is black)
        newly_seen = (valid_zone > 0) & (seen_mask == 0)
        clean_panorama[newly_seen] = warped_frame[newly_seen]

        ## Segmentation
        # Put the current frame on the panorama for the background detection
        pano_input = np.where(valid_zone[..., None] > 0, warped_frame, clean_panorama)
        mask = segmenter.segment(pano_input)
        segmenter.update(pano_input)

        # Increase the mask to detect the border as foreground for a better blending
        mask_dilated = cv2.dilate(mask, np.ones((15, 15), np.uint8))

        # Update
        alpha = np.where((valid_zone > 0) & (seen_mask > 0) & (mask_dilated == 0), 0.05, 0.0)
        alpha_3d = alpha[..., None]
        clean_panorama = (alpha_3d * warped_frame.astype(np.float32)
                       + (1.0 - alpha_3d) * clean_panorama.astype(np.float32)).astype(np.uint8)

        # Update pixels seen in the mask
        seen_mask[valid_zone > 0] = 255

        if display is not None:
            disp_in = cv2.resize(pano_input, (w_img, h_img))
            disp_bg = cv2.resize(clean_panorama, (w_img, h_img))
            display.update(disp_in, disp_bg)

    # Crop the panorama to only keep the part with an actual image
    coords = cv2.findNonZero(seen_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        clean_panorama = clean_panorama[y:y+h, x:x+w]
    
    return clean_panorama

    
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
    parser.add_argument("--phase", type=int)
    args = parser.parse_args()

    frames = load_vid(
        args.video, selected_frames=range(0, args.max_frames, args.step_frames)
    )
    logger.info(f"Number of frames loaded: {len(frames)}")

    # Resize frames
    if args.size is not None:
        frames = resize_frames(frames, tuple(args.size))
    else:
        frames = resize_frames(frames, DEFAULT_SIZE)
    
    if args.phase == 1:
        background, ghost_image = phase1(frames, args.segmentation, args.headless)
        save_output(background, args.output)
        save_output(ghost_image, "output_ghost.png")
    if args.phase == 2 :
        #panorama = stitching_built_in(frames)
        panorama = stitch(frames)
        save_output(panorama, "panorama.png")
    if args.phase == 3: 
        clean_panorama = phase3(frames, args.segmentation, args.headless) 
        save_output(clean_panorama, "clean_panorama.png")
