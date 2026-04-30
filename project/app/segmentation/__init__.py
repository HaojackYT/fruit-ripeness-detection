"""Segmentation package for Fruit Ripeness Detection.
Exports convenience functions for segmentation and visualization.
"""

from .segment import (
    segment_hsv,
    segment_bgr,
    segment_from_bytes,
    mask_to_bboxes,
    get_largest_bbox,
    overlay_mask_on_bgr,
)
from .cli import main as cli_main

__all__ = [
    "segment_hsv",
    "segment_bgr",
    "segment_from_bytes",
    "mask_to_bboxes",
    "get_largest_bbox",
    "overlay_mask_on_bgr",
    "cli_main",
]
