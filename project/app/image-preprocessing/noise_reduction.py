from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np


def _odd_kernel_size(value: int) -> int:
    size = max(1, int(value))
    if size % 2 == 0:
        size += 1
    return size


def apply_gaussian_filter(hsv_img: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Entry function for step 2.4: Gaussian noise reduction.
    """
    cfg = config or {}
    if hsv_img.ndim != 3 or hsv_img.shape[2] != 3:
        raise ValueError("HSV image must have shape HxWx3.")

    hsv = hsv_img.astype(np.float32)
    kernel = _odd_kernel_size(int(cfg.get("gaussian_kernel", 5)))
    sigma = float(cfg.get("gaussian_sigma", 0.0))
    apply_on = str(cfg.get("gaussian_apply_on", "v")).lower()

    if apply_on == "all":
        return cv2.GaussianBlur(hsv, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)

    out = hsv.copy()
    out[:, :, 2] = cv2.GaussianBlur(out[:, :, 2], (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    return out
