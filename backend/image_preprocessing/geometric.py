from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np


def _build_mask(hsv: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    sat_thresh = float(config.get("orientation_sat_thresh", 0.12))
    val_thresh = float(config.get("orientation_val_thresh", 0.12))

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = np.logical_and(sat > sat_thresh, val > val_thresh).astype(np.uint8) * 255

    kernel_size = int(config.get("orientation_morph_kernel", 5))
    kernel_size = max(3, kernel_size | 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _principal_angle_from_mask(mask: np.ndarray, min_points: int) -> Optional[float]:
    ys, xs = np.nonzero(mask)
    if xs.size < min_points:
        return None

    points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    points = points - np.mean(points, axis=0, keepdims=True)

    cov = np.cov(points, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    angle = np.degrees(np.arctan2(principal[1], principal[0]))
    return float(angle)


def _rotation_angle(principal_angle: float, target_axis: str) -> float:
    axis = target_axis.lower()
    if axis == "horizontal":
        return -principal_angle
    return 90.0 - principal_angle


def _rotate_image(img: np.ndarray, angle: float, border_mode: int) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=(0.0, 0.0, 0.0),
    )


def align_orientation_pca(hsv_img: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Entry function for step 2.5: geometric normalization with PCA orientation alignment.
    """
    cfg = config or {}
    if hsv_img.ndim != 3 or hsv_img.shape[2] != 3:
        raise ValueError("HSV image must have shape HxWx3.")

    hsv = hsv_img.astype(np.float32)
    mask = _build_mask(hsv, cfg)

    min_points = int(cfg.get("orientation_min_points", 100))
    principal_angle = _principal_angle_from_mask(mask, min_points=min_points)
    if principal_angle is None:
        return hsv

    target_axis = str(cfg.get("orientation_target_axis", "vertical"))
    rotate_deg = _rotation_angle(principal_angle, target_axis=target_axis)

    border_mode_name = str(cfg.get("orientation_border_mode", "reflect")).lower()
    border_mode = cv2.BORDER_REFLECT if border_mode_name == "reflect" else cv2.BORDER_CONSTANT

    rotated = _rotate_image(hsv, angle=rotate_deg, border_mode=border_mode)
    return rotated.astype(np.float32)
