from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


def _ensure_three_channels(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError("Input image must have shape HxW or HxWx3.")


def _to_bgr(img: np.ndarray, input_color_space: str) -> np.ndarray:
    color_space = input_color_space.lower()
    if color_space == "bgr":
        return img
    if color_space == "rgb":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if color_space == "gray":
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    raise ValueError("input_color_space must be one of: bgr, rgb, gray.")


def _clip_roi(x: int, y: int, w: int, h: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def _extract_roi_with_config(img_bgr: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    height, width = img_bgr.shape[:2]
    roi = config.get("roi")
    if roi is not None:
        if not isinstance(roi, Sequence) or len(roi) != 4:
            raise ValueError("config['roi'] must be a 4-value sequence: (x, y, width, height).")
        x, y, roi_w, roi_h = [int(v) for v in roi]
        x, y, roi_w, roi_h = _clip_roi(x, y, roi_w, roi_h, width, height)
        return img_bgr[y : y + roi_h, x : x + roi_w]

    if not bool(config.get("auto_roi", True)):
        return img_bgr

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat_thresh = int(config.get("auto_roi_sat_thresh", 35))
    val_thresh = int(config.get("auto_roi_val_thresh", 30))
    mask = np.logical_and(hsv[:, :, 1] > sat_thresh, hsv[:, :, 2] > val_thresh).astype(np.uint8) * 255

    kernel_size = int(config.get("auto_roi_morph_kernel", 5))
    kernel_size = max(3, kernel_size | 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    min_area_ratio = float(config.get("auto_roi_min_area_ratio", 0.05))
    if area < min_area_ratio * float(height * width):
        return img_bgr

    x, y, roi_w, roi_h = cv2.boundingRect(largest)
    pad_ratio = float(config.get("roi_padding_ratio", 0.05))
    pad_x = int(roi_w * pad_ratio)
    pad_y = int(roi_h * pad_ratio)
    x, y, roi_w, roi_h = _clip_roi(
        x - pad_x,
        y - pad_y,
        roi_w + (2 * pad_x),
        roi_h + (2 * pad_y),
        width,
        height,
    )
    return img_bgr[y : y + roi_h, x : x + roi_w]


def _normalize_rgb(img_rgb: np.ndarray, normalize: bool) -> np.ndarray:
    rgb = img_rgb.astype(np.float32)
    if normalize:
        rgb = rgb / 255.0
    return rgb


def resize_and_normalize(img: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Entry function for step 2.1: ROI-based resize and normalization.

    Returns:
        RGB image as float32. When normalize=True, value range is [0, 1].
    """
    if img is None:
        raise ValueError("Input image is None.")

    cfg = config or {}
    src = _ensure_three_channels(img)
    input_color_space = str(cfg.get("input_color_space", "bgr")).lower()
    src_bgr = _to_bgr(src, input_color_space)

    roi_img = _extract_roi_with_config(src_bgr, cfg)

    target_size = cfg.get("target_size", (256, 256))
    if not isinstance(target_size, Sequence) or len(target_size) != 2:
        raise ValueError("config['target_size'] must be (width, height).")
    target_w, target_h = [int(v) for v in target_size]
    if target_w <= 0 or target_h <= 0:
        raise ValueError("target_size values must be > 0.")

    resized = cv2.resize(roi_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    normalize = bool(cfg.get("normalize", True))
    return _normalize_rgb(rgb, normalize=normalize)
