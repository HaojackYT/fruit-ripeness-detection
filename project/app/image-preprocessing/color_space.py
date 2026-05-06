from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np


def _prepare_rgb_float(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("RGB input must have shape HxWx3.")

    if np.issubdtype(img.dtype, np.floating):
        rgb = img.astype(np.float32)
        max_value = float(np.max(rgb)) if rgb.size else 1.0
        if max_value > 1.0:
            rgb = rgb / 255.0
    else:
        rgb = img.astype(np.float32) / 255.0

    return np.clip(rgb, 0.0, 1.0)


def rgb_to_hsv(img: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Entry function for step 2.2: RGB to HSV transform.
    OpenCV float HSV uses H in [0, 360], S in [0, 1], V in [0, 1].
    """
    _ = config
    rgb = _prepare_rgb_float(img)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return hsv.astype(np.float32)
