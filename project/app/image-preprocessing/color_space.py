from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np


def _prepare_rgb_float(img: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa ma trận ảnh đầu vào (RGB) có kiểu dữ liệu float32 và các pixel nằm trong khoảng [0, 1].

    Tham số:
        img (np.ndarray): Ma trận ảnh đầu vào.

    Trả về:
        np.ndarray: Ma trận ảnh với kiểu dữ liệu float32 và các pixel nằm trong khoảng [0, 1].

    Ngoại lệ:
        ValueError: Ảnh đầu vào (RGB) không có dạng HxWx3.
    """
    # ndim: số lượng chiều của ảnh
    # shape: tuple chứa chiều cao [0], chiều rộng [1] và số kênh màu [2] 
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Ảnh đầu vào (RGB) phải có dạng HxWx3.")

    # Kiểm tra ảnh đầu vào có phải là số thực (float) hay không
    # np.issubdtype: kiểm tra kiểu dữ liệu có nằm trong hệ thống phân cấp thứ bậc của nó hay không
    if np.issubdtype(img.dtype, np.floating):
        rgb = img.astype(np.float32)
        max_value = float(np.max(rgb)) if rgb.size else 1.0 # Nếu ảnh rỗng (size == 0), đặt max_value là 1.0 để tránh chia cho 0
        # Nếu giá trị lớn nhất trong ảnh > 1 => Giá trị của ảnh nằm trong khoảng [0, 255]
        if max_value > 1.0:
            rgb = rgb / 255.0
    else:
        rgb = img.astype(np.float32) / 255.0

    # Chuẩn hóa giá trị pixel nằm trong khoảng [0, 1]
    return np.clip(rgb, 0.0, 1.0)


def rgb_to_hsv(img: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Hàm entry-point cho bước "2.2 Color Space Transform" trong pipeline "2. Image Preprocessing".

    Tham số:
        img (np.ndarray): Ma trận ảnh đầu vào (RGB) (có thể là uint8 [0, 255] hoặc float [0, 1]).
        config (Optional[Dict[str, Any]], optional): Dictionary chứa các cấu hình cho pipeline. (mặc định là None)

    Trả về:
        np.ndarray: Ma trận ảnh với không gian màu HSV và kiểu dữ liệu float32.
    """
    # _ = config
    rgb = _prepare_rgb_float(img)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Chuẩn hóa kiểu dữ liệu đồng nhất cho bước tiếp theo
    return hsv.astype(np.float32)
