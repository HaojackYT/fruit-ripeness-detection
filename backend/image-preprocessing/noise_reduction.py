from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np


def _odd_kernel_size(value: int) -> int:
    """
    Chuẩn hóa kích thước kernel cho bộ lọc Gaussian là một số lẻ dương (1, 3, 5, ...),
    cho phép tích chập (convolution).

    Tham số:
        value (int): Kích thước kernel đầu vào (có thể là số chẵn hoặc lẻ).

    Trả về:
        int: Kích thước kernel đã được chuẩn hóa.
    """
    # Đảm bảo kích thước kernel là số nguyên dương và tối thiểu là 1 (>= 1)
    size = max(1, int(value))
    # Nếu kích thước kernel là số chẵn, + 1 để trở thành số lẻ
    if size % 2 == 0:
        size += 1
    return size


def apply_gaussian_filter(hsv_img: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Hàm entry-point cho bước "2.4 Noise Reduction" trong pipeline "2. Image Preprocessing".

    Tham số:
        hsv_img (np.ndarray): Ma trận ảnh đầu vào (HSV).
        config (Optional[Dict[str, Any]]): Configuration dictionary for filter tuning. Hỗ trợ các key:
            - 'gaussian_kernel' (int): Kích thước kernel cho bộ lọc Gaussian (mặc định là 5)
            - 'gaussian_sigma' (float): Giá trị sigma cho bộ lọc Gaussian (mặc định là 0.0)
            - 'gaussian_apply_on' (str): Kênh màu mục tiêu để áp dụng bộ lọc Gaussian (mặc định là "v" (kênh Value))

    Trả về:
        np.ndarray: Ma trận ảnh (HSV) cùng kích thước kernel đã được áp dụng bộ lọc Gaussian.
        

    Ngoại lệ:
        ValueError: Ảnh đầu vào (HSV) phải có dạng HxWx3.
    """
    cfg = config or {}
    # ndim: số lượng chiều của ảnh
    # shape: tuple chứa chiều cao [0], chiều rộng [1] và số kênh màu [2] 
    if hsv_img.ndim != 3 or hsv_img.shape[2] != 3:
        raise ValueError("Ảnh đầu vào (HSV) phải có dạng HxWx3.")

    hsv = hsv_img.astype(np.float32)
    # Thiết lập kích thước kernel cho bộ lọc Gaussian (mặc định là 5)
    # TODO
    kernel = _odd_kernel_size(int(cfg.get("gaussian_kernel", 5)))
    # Thiết lập giá trị sigma cho bộ lọc Gaussian (mặc định là 0.0)
    # TODO
    sigma = float(cfg.get("gaussian_sigma", 0.0))
    # Thiết lập kênh màu mục tiêu để áp dụng bộ lọc Gaussian (mặc định là "v" (kênh Value))
    apply_on = str(cfg.get("gaussian_apply_on", "v")).lower()

    if apply_on == "all":
        return cv2.GaussianBlur(hsv, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)

    out = hsv.copy()
    out[:, :, 2] = cv2.GaussianBlur(out[:, :, 2], (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    return out
