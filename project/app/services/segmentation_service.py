from __future__ import annotations

import base64
from typing import Any, Dict, Optional

import cv2
import numpy as np

from app.segmentation import (
    get_largest_bbox,
    overlay_mask_on_bgr,
    segment_from_bytes,
)


def _decode_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img


def _encode_jpeg_data_url(bgr_img: np.ndarray, quality: int = 90) -> str:
    success, encoded = cv2.imencode(
        ".jpg",
        bgr_img,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not success:
        raise ValueError("Failed to encode overlay image")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def segment_image(image_bytes: bytes) -> Dict[str, Any]:
    """Run HSV segmentation and return overlay + basic stats."""
    bgr = _decode_bgr(image_bytes)
    mask = segment_from_bytes(image_bytes)

    overlay = overlay_mask_on_bgr(bgr, mask, color=(0, 255, 0), alpha=0.45)
    overlay_url = _encode_jpeg_data_url(overlay)

    mask_area = int(np.count_nonzero(mask))
    image_area = int(mask.shape[0] * mask.shape[1])
    mask_ratio = (mask_area / image_area) if image_area else 0.0

    bbox = get_largest_bbox(mask, min_area=0)
    bbox_dict: Optional[Dict[str, int]] = None
    if bbox:
        x, y, w, h = bbox
        bbox_dict = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

    return {
        "overlay_image": overlay_url,
        "mask_ratio": round(mask_ratio, 4),
        "mask_area": mask_area,
        "image_area": image_area,
        "bbox": bbox_dict,
    }
