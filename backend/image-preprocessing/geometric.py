from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np


def _build_mask(hsv: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    Tạo mặt nạ nhị phân (binary mask) để tách vùng chứa quả khỏi nền ảnh .

    Tham số:
        hsv (np.ndarray): Ma trận ảnh đầu vào (HSV).
        config (Dict[str, Any]): Dictionary chứa cấu hình. Hỗ trợ các key:
            - 'orientation_sat_thresh': Ngưỡng độ bão hòa (mặc định 0.12).
            - 'orientation_val_thresh': Ngưỡng độ sáng (mặc định 0.12).
            - 'orientation_morph_kernel': Kích thước Structuring Element (kernel) (mặc định 5).

    Trả về:
        np.ndarray: Ma trận đen trắng (đen = nền, trắng = vùng chứa quả) có cùng kích thước với ma trận ảnh đầu vào.
    """
    # Thiếp lập ngưỡng độ bão hòa và độ sáng (mặc định là 0.12)
    sat_thresh = float(config.get("orientation_sat_thresh", 0.12))
    val_thresh = float(config.get("orientation_val_thresh", 0.12))

    sat = hsv[:, :, 1] # kênh Saturation
    val = hsv[:, :, 2] # kênh Value
    # logical_and: trả về True chỉ cho pixel vừa có màu sắc đậm (độ bão hòa cao) vừa có độ sáng cao
    # => pixel có khả năng thuộc về quả, ngược lại là nền hoặc bóng râm
    # astype(np.uint8) * 255: chuyển đổi từ ma trận boolean (True/False) -> ma trận nhị phân (0/1) -> ma trận đen trắng (0/255)
    mask = np.logical_and(sat > sat_thresh, val > val_thresh).astype(np.uint8) * 255

    # Thiết lập kích thước Structuring Element (kernel) (mặc định là 5)
    # TODO
    kernel_size = int(config.get("orientation_morph_kernel", 5))
    # Đảm bảo Structuring Element là ma trận lẻ và tối thiểu là 3
    # kernel_size | 1: bit cuối cùng bên phải của kernel_size thành 1
    # Số chẵn bit cuối cùng bên phải là 0
    # Số lẻ bit cuối cùng bên phải sẽ là 1
    kernel_size = max(3, kernel_size | 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    # Morphological Opening (Erosion -> Dilation)
    # Loại bỏ các đốm trắng nhỏ (nhiễu muối) ở ngoài nền đen (background)
    # iterations của Morphological Opening > Morphological Closing do nhiễu muối thường nhỏ hơn nhiễu tiêu và dễ bị loại bỏ hơn
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # Morphological Closing (Dilation -> Erosion)
    # Lấp đầy các lỗ đen (nhiễu tiêu) nằm bên trong khối màu trắng của trái (foreground)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _principal_angle_from_mask(mask: np.ndarray, min_points: int) -> Optional[float]:
    """
    Tính toán góc nghiêng của trục chính (eigenvector của eigenvalue lớn nhất) của quả dựa trên PCA.

    Tham số:
        mask (np.ndarray): Ma trận nhị phân của quả.
        min_points (int): Số lượng pixel tối thiểu để thực hiện PCA.

    Trả về:
        Optional[float]: Góc của trục chính tính bằng độ (-180 đến 180).
        None nếu số lượng pixel trên mask < min_points.
    """
    # np.nonzero: trả về tuple chứa 2 mảng 1 chiều (số lượng mảng = số lượng chiều của ma trận đầu vào)
    # duyệt lần lượt theo từng hàng, tọa độ y và x của các pixel có giá trị khác 0
    ys, xs = np.nonzero(mask)
    # Kiểm tra số lượng pixel trên mask để thực hiện tính toán PCA
    if xs.size < min_points:
        return None

    # column_stack: xếp các mảng 1 chiều thành một ma trận 2 chiều theo cột (số lượng cột = số lượng mảng đầu vào)
    points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    # np.mean: tính trung bình cộng của các điểm tọa độ (x, y) trong mảng
    # variance: khoảng cách bình phương trung bình của các điểm so với mean
    # cho thấy các điểm được biễu diễn như thế nào khi so với mean
    # centering: các điểm tọa độ (x, y) trừ đi mean để tâm của đám mây biểu diễn các điểm tọa độ về gốc tọa độ (0, 0)
    # giúp PCA tập trung hoàn toàn vào variance thay vì vị trí tuyệt đối của các điểm tọa độ
    points = points - np.mean(points, axis=0, keepdims=True)

    # covarience: đo lường variance biến thiên (cùng tăng hoặc cùng giảm) giữa hai trục x, y
    # covarience matrix: ma trận 2x2 chứa các giá trị variance, covariance của 2 điểm tọa độ
    # để biểu diễn sự phân bố của 2 điểm tọa độ trong không gian 2 chiều
    # [ variance_x, covariance_xy
    #   covariance_yx, variance_y ]
    # Đường chéo chính (main diagonal): biểu diễn variance theo từng trục x, y.
    # Giá trị nào lớn hơn chứng tỏ chiều (ngang/dọc) phân tán mạnh hơn (chứa nhiều thông tin hơn)
    # Đường chéo phụ (off-diagonal): biểu diễn covariance giữa 2 điểm tọa độ
    # Giá trị != 0: đám mây biểu diễn các điểm tọa độ nằm nghiêng 
    # Giá trị = 0: đám mây biểu diễn các điểm tọa độ thẳng đứng, trục x và y không liên quan đến nhau
    cov = np.cov(points, rowvar=False)
    # np.linalg.eigh: dựa vào spectral theorem để tính toán eigenvalues và eigenvectors của covariance matrix,
    # trả về mỗi cột là một eigenvector riêng
    # spectral theorem: eigenvalues của chúng luôn là số thực (không có phần ảo) và eigenvector của chúng vuông góc với nhau.
    # eigenvalue: một giá trị vô hướng chỉ ra độ lớn (ảnh hưởng đến việc phân bố của dữ liệu)
    # của variance dọc theo hướng của eigenvector
    # eigenvector: chỉ ra các hướng mà dữ liệu phân tán
    eigvals, eigvecs = np.linalg.eigh(cov)
    # np.argmax: trả về chỉ số của giá trị lớn nhất trong mảng
    # eigvecs[:, np.argmax(eigvals)]: lấy tất cả các hàng của cột tương ứng với eigenvalue lớn nhất để tìm ra vector hướng dọc theo chiều dài nhất của quả
    # Mảng chứa 2 phần tử [x, y] đại diện cho vector hướng dọc theo chiều dài nhất của quả
    principal = eigvecs[:, np.argmax(eigvals)]
    # np.arctan2: tính bằng radians arctan(y/x) trả về góc giữa vector hướng dọc theo chiều dài nhất của quả và trục Ox dương
    # np.degrees: chuyển đổi góc từ radians sang độ (x 180/pi)
    angle = np.degrees(np.arctan2(principal[1], principal[0]))
    return float(angle)


def _rotation_angle(principal_angle: float, target_axis: str) -> float:
    """
    Xác định góc nghiêng cần xoay để đưa trái cây về trục mục tiêu (Ox hoặc Oy).

    Tham số:
        principal_angle (float): Góc nghiêng hiện tại của trái cây (đã tính qua PCA).
        target_axis (str): Trục mục tiêu ("horizontal": trục ngang Ox hoặc "vertical": trục dọc Oy).

    Trả về:
        float: Góc nghiêng cần truyền vào để căn chỉnh hướng về trục mục tiêu.
    """
    axis = target_axis.lower()
    # Nếu trục mục tiêu là trục ngang Ox, trả về góc âm của góc PCA
    if axis == "horizontal":
        return -principal_angle
    # Nếu trục mục tiêu là trục dọc Oy, trả về góc bù của góc PCA
    return 90.0 - principal_angle


def _rotate_image(img: np.ndarray, angle: float, border_mode: int) -> np.ndarray:
    """
    Thực hiện phép biến đổi Affine để xoay ma trận ảnh quanh tâm ảnh.

    Tham số:
        img (np.ndarray): Ma trận ảnh cần xoay.
        angle (float): Góc xoay tính bằng độ (kết quả của hàm _rotation_angle).
        border_mode (int): Xác định cách xử lý các pixel viền bị trống sau khi xoay
        (vd: cv2.BORDER_REFLECT, cv2.BORDER_CONSTANT).

    Trả về:
        np.ndarray: Ma trận ảnh đã được xoay.
    """
    # shape: tuple chứa chiều cao [0], chiều rộng [1] và số kênh màu [2]
    # [:2]: trả về chỉ số [0, 2)
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    # cv2.getRotationMatrix2D: trả về ma trận Affine 2x3
    # TODO
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # cv2.warpAffine: áp dụng ma trận Affine để xoay ảnh, giữ nguyên kích thước gốc (w, h)
    # TODO
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
    Hàm entry-point cho bước "2.5 Geometric Normalization" trong pipeline "2. Image Preprocessing".

    Tham số:
        hsv_img (np.ndarray): Ma trận ảnh đầu vào (HSV).
        config (Optional[Dict[str, Any]]): Cấu hình tùy chọn chứa các tham số chạy thuật toán.

    Trả về:
        np.ndarray: Ma trận ảnh (HSV) đã được xoay.

    Ngoại lệ:
        ValueError: Ma trận ảnh đầu vào (HSV) phải có dạng HxWx3.
    """
    cfg = config or {}
    if hsv_img.ndim != 3 or hsv_img.shape[2] != 3:
        raise ValueError("Ma trận ảnh đầu vào (HSV) phải có dạng HxWx3.")

    hsv = hsv_img.astype(np.float32)
    mask = _build_mask(hsv, cfg)

    # Thiết lập số lượng pixel tối thiểu để thực hiện PCA (mặc định là 100)
    min_points = int(cfg.get("orientation_min_points", 100))
    principal_angle = _principal_angle_from_mask(mask, min_points=min_points)
    # Nếu số lượng pixel trên mask không đủ để thực hiện PCA, trả về ảnh gốc mà không xoay
    if principal_angle is None:
        return hsv

    # Thiết lập trục mục tiêu để căn chỉnh hướng về trục mục tiêu (mặc định là "vertical")
    # TODO
    target_axis = str(cfg.get("orientation_target_axis", "vertical"))
    rotate_deg = _rotation_angle(principal_angle, target_axis=target_axis)

    # Thiết lập cách xử lý các pixel viền bị trống sau khi xoay (mặc định là "reflect")
    # TODO
    border_mode_name = str(cfg.get("orientation_border_mode", "reflect")).lower()
    border_mode = cv2.BORDER_REFLECT if border_mode_name == "reflect" else cv2.BORDER_CONSTANT

    rotated = _rotate_image(hsv, angle=rotate_deg, border_mode=border_mode)
    return rotated.astype(np.float32)
