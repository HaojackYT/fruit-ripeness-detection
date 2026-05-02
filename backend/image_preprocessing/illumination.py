from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _validate_hsv(hsv_img: np.ndarray) -> np.ndarray:
    if hsv_img.ndim != 3 or hsv_img.shape[2] != 3:
        raise ValueError("HSV image must have shape HxWx3.")
    return hsv_img.astype(np.float32)


def _estimate_gamma(v_channel: np.ndarray, target_mean: float) -> float:
    target = float(np.clip(target_mean, 0.05, 0.95))
    current = float(np.clip(np.mean(v_channel), 1e-6, 1.0 - 1e-6))
    gamma = np.log(target) / np.log(current)
    return float(np.clip(gamma, 0.35, 2.5))


def gamma_correction(hsv_img: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Entry function for step 2.3: illumination normalization via gamma correction.
    """
    cfg = config or {}
    hsv = _validate_hsv(hsv_img)
    out = hsv.copy()

    v = np.clip(out[:, :, 2], 0.0, 1.0)

    if bool(cfg.get("auto_gamma", False)):
        gamma = _estimate_gamma(v, target_mean=float(cfg.get("target_mean_v", 0.55)))
    else:
        gamma = float(cfg.get("gamma", 1.0))

    if gamma <= 0.0:
        raise ValueError("Gamma must be > 0.")

    # gamma < 1 brightens, gamma > 1 darkens
    corrected_v = np.power(v, gamma).astype(np.float32)
    out[:, :, 2] = np.clip(corrected_v, 0.0, 1.0)
    return out
