from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _validate_hsv(hsv_img: np.ndarray) -> np.ndarray:
    """
    Xác thực và chuẩn hóa ảnh đầu vào cho pipeline.

    Tham số:
        hsv_img (np.ndarray): Ma trận ảnh đầu vào (HSV).

    Trả về:
        np.ndarray: Ma trận ảnh đã được xác thực và chuẩn hóa.

    Ngoại lệ:
        ValueError: Ảnh đầu vào (HSV) phải có dạng HxWx3.
    """
    # ndim: số lượng chiều của ảnh
    # shape: tuple chứa chiều cao [0], chiều rộng [1] và số kênh màu [2] 
    if hsv_img.ndim != 3 or hsv_img.shape[2] != 3:
        raise ValueError("Ảnh đầu vào (HSV) phải có dạng HxWx3.")
    # Chuẩn hóa kiểu dữ liệu đồng nhất cho bước tiếp theo
    return hsv_img.astype(np.float32)


def _estimate_gamma(v_channel: np.ndarray, target_mean: float) -> float:
    """
    Ước lượng giá trị gamma cần thiết để kéo độ sáng trung bình của ảnh về target_mean,
    giúp các đặc trưng màu sắc đồng đều hơn trước khi đưa vào SVM.

    Tham số:
        v_channel (np.ndarray): Kênh Value được tách ra từ ảnh HSV. (đã chuẩn hóa bằng _validate_hsv).
        target_mean (float): Mức độ sáng trung bình của ảnh mong muốn (mặc định là 0.55).

    Trả về:
        float: Hệ số gamma đã được tính toán (giới hạn trong khoảng [0.35, 2.5]).
    """
    target = float(np.clip(target_mean, 0.05, 0.95))
    # np.mean: tính trung bình cộng của các tọa độ (x, y) trong mảng
    # mean = 0 (ảnh tối) => log(0) = âm vô cùng
    # mean = 1 (ảnh sáng) => log(1) = 0 => gamma = log(target) / 0 => không xác định
    current = float(np.clip(np.mean(v_channel), 1e-6, 1.0 - 1e-6))
    # target = current ^ gamma
    # <=> log(target) = gamma * log(current)
    # => gamma = log(target) / log(current)
    gamma = np.log(target) / np.log(current)
    # Giới hạn giá trị gamma trong một khoảng hợp lý để tránh việc thuật toán chỉnh sửa ảnh quá mức
    # TODO
    return float(np.clip(gamma, 0.35, 2.5))


def gamma_correction(hsv_img: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Hàm entry-point cho bước "2.3 Illumination Normalization" trong pipeline "2. Image Preprocessing".

    Tham số:
        hsv_img (np.ndarray): Ma trận ảnh đầu vào (HSV) cần chuẩn hóa ánh sáng.
        config (Optional[Dict[str, Any]]): Dictionary chứa các cấu hình cho thuật toán. Hỗ trợ các key:
            - "auto_gamma" (bool): Nếu True, tự động tính gamma dựa theo độ sáng ảnh (mặc định là False).
            - "target_mean_v" (float): Độ sáng mục tiêu nếu dùng auto_gamma (mặc định 0.55).
            - "gamma" (float): Giá trị gamma tĩnh nếu không dùng auto_gamma (mặc định 1.0).

    Trả về:
        np.ndarray: Ảnh HSV đã được chuẩn hóa độ sáng, sẵn sàng cho bước Segmentation (Bước 3).

    Ngoại lệ:
        ValueError: Nếu cấu hình hệ số gamma <= 0 (gây lỗi toán học khi tính lũy thừa).
    """
    cfg = config or {}
    hsv = _validate_hsv(hsv_img)
    out = hsv.copy()

    # out[:, :, 2]: kênh Value
    # Đảm bảo giá trị của kênh Value nằm trong khoảng [0, 1] trước khi áp dụng gamma correction
    v = np.clip(out[:, :, 2], 0.0, 1.0)

    # Kiểm tra auto_gamma có được kích hoạt hay không (mặc định là False)
    if bool(cfg.get("auto_gamma", False)):
        # Ước lượng gamma dựa trên độ sáng của kênh Value và độ sáng mục tiêu (mặc định là 0.55)
        # TODO
        gamma = _estimate_gamma(v, target_mean=float(cfg.get("target_mean_v", 0.55)))
    else:
        # gamma tĩnh (mặc định là 1.0) không làm thay đổi độ sáng của ảnh
        gamma = float(cfg.get("gamma", 1.0))

    if gamma <= 0.0:
        raise ValueError("Gamma phải > 0.")

    # gamma < 1 sáng hơn, gamma > 1 tối hơn
    # target = current ^ gamma
    corrected_v = np.power(v, gamma).astype(np.float32)
    # Đảm bảo giá trị của kênh Value nằm trong khoảng [0, 1] sau khi áp dụng gamma correction
    out[:, :, 2] = np.clip(corrected_v, 0.0, 1.0)
    return out
