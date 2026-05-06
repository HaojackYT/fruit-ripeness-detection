"""Core segmentation utilities.

Functions:
- segment_hsv: segment from HSV float image (H in degrees 0..360, S,V in [0,1])
- segment_bgr: accept BGR(u8) images and run segmentation
- segment_from_bytes: decode bytes then segment
- mask_to_bboxes, get_largest_bbox: helpers to extract bounding boxes
- overlay_mask_on_bgr: produce visualization overlay

This module is intentionally self-contained and does not require the preprocessing
package to be importable (works with raw BGR images or HSV float arrays).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2


DEFAULT_SEGMENT_CONFIG: Dict[str, Any] = {
    # thresholds expect HSV float where H in degrees [0,360], S and V in [0,1]
    "sat_thresh": 0.12,
    "val_thresh": 0.12,
    # optional list of (h_min, h_max) hue ranges in degrees; supports wrap-around
    "hue_ranges": None,
    "morph_kernel": 5,
    "morph_open_iters": 1,
    "morph_close_iters": 2,
    "min_area": 500,
}


def _odd_kernel_size(value: int) -> int:
    k = max(1, int(value))
    if k % 2 == 0:
        k += 1
    return k


def _hsv_from_bgr(bgr_img: np.ndarray) -> np.ndarray:
    if bgr_img is None:
        raise ValueError("Input image is None")
    if bgr_img.ndim != 3 or bgr_img.shape[2] != 3:
        raise ValueError("BGR input must be HxWx3")

    if bgr_img.dtype == np.uint8:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_f = rgb.astype(np.float32) / 255.0
        hsv = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2HSV)
        return hsv.astype(np.float32)

    if np.issubdtype(bgr_img.dtype, np.floating):
        # assume it's RGB float in [0,1]
        arr = bgr_img.astype(np.float32)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        return hsv.astype(np.float32)

    raise ValueError("Unsupported image dtype for conversion to HSV")


def _ensure_hsv_float(hsv_img: np.ndarray) -> np.ndarray:
    if hsv_img is None:
        raise ValueError("Input image is None")
    if hsv_img.ndim != 3 or hsv_img.shape[2] != 3:
        raise ValueError("HSV image must have shape HxWx3")

    hsv = hsv_img.astype(np.float32).copy()

    # If HSV appears scaled with H in [0,1], expand it to degrees
    if float(np.max(hsv[:, :, 0])) <= 1.0:
        hsv[:, :, 0] = hsv[:, :, 0] * 360.0

    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0.0, 1.0)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0.0, 1.0)

    return hsv


def _apply_hue_filters(h: np.ndarray, hue_ranges: Optional[List[Tuple[float, float]]]) -> np.ndarray:
    if not hue_ranges:
        return np.ones(h.shape, dtype=bool)

    mask = np.zeros(h.shape, dtype=bool)
    for (h_min, h_max) in hue_ranges:
        # normalize to [0,360)
        h_min = float(h_min) % 360.0
        h_max = float(h_max) % 360.0
        if h_min <= h_max:
            mask |= (h >= h_min) & (h <= h_max)
        else:
            # wrap-around (e.g., 350..10)
            mask |= (h >= h_min) | (h <= h_max)
    return mask


def segment_hsv(hsv_img: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Segment an HSV float image.

    Returns a uint8 mask (0/255) of the segmented object(s).
    """
    cfg = DEFAULT_SEGMENT_CONFIG.copy()
    if config:
        cfg.update(config)

    hsv = _ensure_hsv_float(hsv_img)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    sat_thresh = float(cfg.get("sat_thresh", 0.12))
    val_thresh = float(cfg.get("val_thresh", 0.12))

    mask = (s > sat_thresh) & (v > val_thresh)

    hue_ranges = cfg.get("hue_ranges")
    if hue_ranges:
        hue_mask = _apply_hue_filters(h, hue_ranges)
        mask &= hue_mask

    mask_u8 = (mask.astype(np.uint8)) * 255

    # morphological cleanup
    k = _odd_kernel_size(int(cfg.get("morph_kernel", 5)))
    kernel = np.ones((k, k), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=int(cfg.get("morph_open_iters", 1)))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=int(cfg.get("morph_close_iters", 2)))

    # remove small regions
    min_area = int(cfg.get("min_area", 0))
    if min_area > 0:
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(mask_u8)
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                cv2.drawContours(filtered, [c], -1, 255, thickness=cv2.FILLED)
        mask_u8 = filtered

    return mask_u8


def segment_bgr(bgr_img: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    hsv = _hsv_from_bgr(bgr_img)
    return segment_hsv(hsv, config=config)


def segment_from_bytes(image_bytes: bytes, flags: int = cv2.IMREAD_COLOR, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, flags)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return segment_bgr(img, config=config)


def mask_to_bboxes(mask: np.ndarray, min_area: int = 0) -> List[Tuple[int, int, int, int]]:
    if mask is None:
        return []
    mask_u8 = mask.copy().astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))
    # sort by area descending
    boxes_sorted = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    return boxes_sorted


def get_largest_bbox(mask: np.ndarray, min_area: int = 0) -> Optional[Tuple[int, int, int, int]]:
    boxes = mask_to_bboxes(mask, min_area=min_area)
    if not boxes:
        return None
    return boxes[0]


def overlay_mask_on_bgr(bgr_img: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.5) -> np.ndarray:
    if bgr_img is None or mask is None:
        raise ValueError("bgr_img and mask are required")
    if bgr_img.ndim != 3 or bgr_img.shape[2] != 3:
        raise ValueError("bgr_img must be HxWx3")
    if mask.shape[:2] != bgr_img.shape[:2]:
        raise ValueError("mask and image must share spatial dimensions")

    overlay = bgr_img.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32).reshape((1, 1, 3))
    mask_bool = (mask > 0)
    overlay[mask_bool] = overlay[mask_bool] * (1.0 - alpha) + color_arr * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)
