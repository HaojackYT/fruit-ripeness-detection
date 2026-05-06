from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

try:
    from .color_space import rgb_to_hsv
    from .geometric import align_orientation_pca
    from .illumination import gamma_correction
    from .noise_reduction import apply_gaussian_filter
    from .resize_normalize import resize_and_normalize
except ImportError:
    from color_space import rgb_to_hsv
    from geometric import align_orientation_pca
    from illumination import gamma_correction
    from noise_reduction import apply_gaussian_filter
    from resize_normalize import resize_and_normalize


DEFAULT_PREPROCESS_CONFIG: Dict[str, Any] = {
    "input_color_space": "bgr",
    "roi": None,
    "auto_roi": True,
    "auto_roi_sat_thresh": 35,
    "auto_roi_val_thresh": 30,
    "auto_roi_morph_kernel": 5,
    "auto_roi_min_area_ratio": 0.05,
    "roi_padding_ratio": 0.05,
    "target_size": (256, 256),
    "normalize": True,
    "gamma": 1.0,
    "auto_gamma": True,
    "target_mean_v": 0.55,
    "gaussian_kernel": 5,
    "gaussian_sigma": 0.0,
    "gaussian_apply_on": "v",
    "orientation_sat_thresh": 0.12,
    "orientation_val_thresh": 0.12,
    "orientation_morph_kernel": 5,
    "orientation_min_points": 100,
    "orientation_target_axis": "vertical",
    "orientation_border_mode": "reflect",
    "return_intermediate": False,
}


def _merge_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = DEFAULT_PREPROCESS_CONFIG.copy()
    if config:
        merged.update(config)
    return merged


def preprocess_image(
    img: np.ndarray, config: Optional[Dict[str, Any]] = None
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Orchestrator for module 2. Image Preprocessing.

    Pipeline:
    2.1 ROI-based Resize + Normalize -> RGB image
    2.2 RGB -> HSV
    2.3 Gamma correction on V channel
    2.4 Gaussian filter
    2.5 PCA-based orientation alignment
    """
    cfg = _merge_config(config)

    step_21 = resize_and_normalize(img, cfg)
    step_22 = rgb_to_hsv(step_21, cfg)
    step_23 = gamma_correction(step_22, cfg)
    step_24 = apply_gaussian_filter(step_23, cfg)
    step_25 = align_orientation_pca(step_24, cfg)

    if bool(cfg.get("return_intermediate", False)):
        return {
            "resize_normalize": step_21,
            "color_space": step_22,
            "illumination": step_23,
            "noise_reduction": step_24,
            "geometric": step_25,
            "output": step_25,
        }

    return step_25


def _hsv_to_bgr_u8(hsv_img: np.ndarray) -> np.ndarray:
    hsv = hsv_img.astype(np.float32).copy()
    if float(np.max(hsv[:, :, 0])) <= 1.0:
        hsv[:, :, 0] = hsv[:, :, 0] * 360.0

    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0.0, 1.0)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0.0, 1.0)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb_u8 = (rgb * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)


def _load_json_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object (dictionary).")
    return data


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline for Fruit Ripeness Detection.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--output",
        default="preprocessed_output.jpg",
        help="Path to save visualization output (BGR JPG/PNG).",
    )
    parser.add_argument("--config-json", default=None, help="Optional JSON config file.")
    parser.add_argument("--width", type=int, default=None, help="Target resize width.")
    parser.add_argument("--height", type=int, default=None, help="Target resize height.")
    parser.add_argument("--gamma", type=float, default=None, help="Gamma value for illumination normalization.")
    parser.add_argument("--auto-gamma", action="store_true", help="Enable auto gamma estimation.")
    parser.add_argument("--no-auto-roi", action="store_true", help="Disable automatic ROI extraction.")
    parser.add_argument(
        "--save-hsv-npy",
        default=None,
        help="Optional .npy file to save final HSV float output for downstream segmentation.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        parser.error(f"Image does not exist: {img_path}")

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        parser.error(f"Cannot read image: {img_path}")

    user_config = _load_json_config(args.config_json)

    if (args.width is None) != (args.height is None):
        parser.error("Please provide both --width and --height together.")

    if args.width is not None and args.height is not None:
        user_config["target_size"] = (args.width, args.height)

    if args.gamma is not None:
        user_config["gamma"] = args.gamma

    if args.auto_gamma:
        user_config["auto_gamma"] = True

    if args.no_auto_roi:
        user_config["auto_roi"] = False

    output = preprocess_image(img, user_config)
    if isinstance(output, dict):
        hsv_final = output["output"]
    else:
        hsv_final = output

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vis_bgr = _hsv_to_bgr_u8(hsv_final)
    ok = cv2.imwrite(str(out_path), vis_bgr)
    if not ok:
        parser.error(f"Cannot write output image: {out_path}")

    if args.save_hsv_npy:
        np.save(args.save_hsv_npy, hsv_final)

    print("Preprocessing completed successfully.")
    print(f"Input: {img_path}")
    print(f"Output visualization: {out_path}")
    if args.save_hsv_npy:
        print(f"Saved HSV tensor: {args.save_hsv_npy}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
