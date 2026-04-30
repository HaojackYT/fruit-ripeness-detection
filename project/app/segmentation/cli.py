"""Command-line runner for segmentation module."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from .segment import segment_bgr, overlay_mask_on_bgr


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run simple segmentation for fruit images")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--output-mask", default=None, help="Path to save mask (PNG)")
    p.add_argument("--output-overlay", default=None, help="Path to save overlay visualization (JPG/PNG)")
    p.add_argument("--config-json", default=None, help="Optional segmentation config (JSON file)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    img_path = Path(args.image)
    if not img_path.exists():
        parser.error(f"Image does not exist: {img_path}")

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        parser.error(f"Cannot read image: {img_path}")

    config: Dict[str, Any] = {}
    if args.config_json:
        cfg_path = Path(args.config_json)
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                config.update(data)

    mask = segment_bgr(img, config=config)

    if args.output_mask:
        out_mask = Path(args.output_mask)
        out_mask.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_mask), mask)

    if args.output_overlay:
        out_overlay = Path(args.output_overlay)
        out_overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay = overlay_mask_on_bgr(img, mask)
        cv2.imwrite(str(out_overlay), overlay)

    print("Segmentation completed.")
    if args.output_mask:
        print(f"Saved mask: {args.output_mask}")
    if args.output_overlay:
        print(f"Saved overlay: {args.output_overlay}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
