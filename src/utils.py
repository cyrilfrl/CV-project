
from pathlib import Path

import cv2


def _get_parent_dir() -> Path:
    curr_path = Path(__file__).resolve()
    for directory in curr_path.parents:
            if (directory / "pyproject.toml").is_file():
                return directory
    raise FileNotFoundError("Could not find the parent directory containing pyproject.toml")

def resize_frames(frames, size=(960, 540)):
    return [cv2.resize(frame, size) for frame in frames]

def display_frame(frame, mask):
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)


        model.update_and_display(frame)
        print(f"Processed frame {idx + 1}/{len(frames)}")