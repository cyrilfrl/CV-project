import numpy as np


def exponential(B, frame, mask, alpha=0.05):
    """Running average over all pixels."""
    return alpha * frame + (1 - alpha) * B

def selective(B, frame, mask, alpha=0.05):
    """Running average over background pixels only."""
    out = B.copy()
    keep = mask == 0
    out[keep] = alpha * frame[keep] + (1 - alpha) * B[keep]
    return out

def median(B, frame, mask, buffer=None):
    """Median of the buffered frames."""
    return np.median(np.stack(buffer), axis=0)

UPDATERS = {"EXP": exponential, "SELECTIVE": selective, "MEDIAN": median}
