from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


def _ensure_three_channels(img: np.ndarray) -> np.ndarray:
    """
    Đảm bảo ma trận ảnh đầu vào có đúng 3 kênh màu (BGR).

    Tham số:
        img (np.ndarray): Ma trận ảnh đầu vào (có thể là ảnh xám (HxW) hoặc ảnh màu BGR (HxWx3)).

    Trả về:
        np.ndarray: Ma trận ảnh 3 chiều (HxWx3).

    Ngoại lệ:
        ValueError: Nếu ảnh đầu vào không phải là ảnh xám (HxW) hoặc ảnh màu BGR (HxWx3).
    """
    # ndim: số lượng chiều của ảnh
    if img.ndim == 2: # ảnh xám
        # cvtColor: chuyển đổi không gian màu
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # shape: tuple chứa chiều cao [0], chiều rộng [1] và số kênh màu [2]
    if img.ndim == 3 and img.shape[2] == 3: # ảnh màu (BGR)
        return img
    raise ValueError("Ảnh đầu vào phải là ảnh xám (HxW) hoặc ảnh màu BGR (HxWx3).")


def _to_bgr(img: np.ndarray, input_color_space: str) -> np.ndarray:
    """
    Chuyển đổi không gian màu của ảnh đầu vào về định dạng chuẩn BGR của OpenCV.

    Tham số:
        img (np.ndarray): Ma trận ảnh đầu vào (đã đảm bảo 3 kênh bằng _ensure_three_channels).
        input_color_space (str): Không gian màu hiện tại của ảnh đầu vào ('bgr', 'rgb', hoặc 'gray').

    Trả về:
        np.ndarray: Ma trận ảnh với không gian màu BGR.

    Ngoại lệ:
        ValueError: Nếu tham số input_color_space không được hỗ trợ.
    """
    color_space = input_color_space.lower()
    if color_space == "bgr":
        return img
    if color_space == "rgb":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if color_space == "gray":
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    raise ValueError("input_color_space phải là một trong các giá trị: bgr, rgb, gray.")


def _clip_roi(x: int, y: int, w: int, h: int, width: int, height: int) -> Tuple[int, int, int, int]:
    """
    Cắt xén (clip) tọa độ bounding box để đảm bảo không vượt quá kích thước thực tế của ảnh.

    Tham số:
        x (int): Tọa độ X góc trên bên trái của bounding box.
        y (int): Tọa độ Y góc trên bên trái của bounding box.
        w (int): Chiều rộng của bounding box.
        h (int): Chiều cao của bounding box.
        width (int): Chiều rộng tối đa của ảnh gốc.
        height (int): Chiều cao tối đa của ảnh gốc.

    Trả về:
        Tuple[int, int, int, int]: Tọa độ (x, y, w, h) đã được hiệu chỉnh nằm gọn trong ảnh.
    """
    # Đảm bảo tọa độ x >= 0 và không vượt quá chiều rộng của ảnh
    x = max(0, min(x, width - 1))
    # Đảm bảo tọa độ y >= 0 và không vượt quá chiều cao của ảnh
    y = max(0, min(y, height - 1))
    # Đảm bảo chiều rộng của bounding box >= 1 và không vượt quá phần không gian còn lại tính từ x
    w = max(1, min(w, width - x))
    # Đảm bảo chiều cao của bounding box >= 1 và không vượt quá phần không gian còn lại tính từ y
    h = max(1, min(h, height - y))
    return x, y, w, h


def _extract_roi_with_config(img_bgr: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    Trích xuất Vùng quan tâm (ROI - Region of Interest) chứa quả từ ảnh gốc.
    
    Quy trình:
    1. Nếu config cung cấp sẵn 'roi', cắt chính xác theo tọa độ đó.
    2. Nếu không có 'roi' và 'auto_roi' = True, áp dụng thuật toán phân đoạn dựa trên
    không gian HSV để bóc tách quả (táo/xoài/cà chua) khỏi nền.
    3. Trả về toàn bộ ảnh gốc nếu các điều kiện trên không thỏa mãn hoặc không tìm thấy quả.

    Tham số:
        img_bgr (np.ndarray): Ma trận ảnh đầu vào (BGR).
        config (Dict[str, Any]): Cấu hình pipeline. Hỗ trợ các key: 
            - roi (Sequence[int]): [x, y, w, h] các tham số của bounding box.
            - auto_roi (bool): Kích hoạt tự động tìm ROI (mặc định True).
            - auto_roi_sat_thresh (int): Ngưỡng tối thiểu cho độ bão hòa để nhận diện quả (mặc định 35).
            - auto_roi_val_thresh (int): Ngưỡng tối thiểu cho độ sáng để nhận diện quả (mặc định 30).
            - auto_roi_morph_kernel (int): Kích thước Structuring Element (kernel) (mặc định 5).
            - auto_roi_min_area_ratio (float): Tỷ lệ diện tích đường viền tối thiểu để nhận diện là quả (mặc định 0.05).
            - roi_padding_ratio (float): Tỷ lệ padding bounding box sau khi tìm thấy (mặc định 0.05).

    Trả về:
        np.ndarray: Ảnh con (crop) chỉ chứa vùng ROI.

    Ngoại lệ:
        ValueError: Nếu định dạng tham số 'roi' truyền vào không hợp lệ.
    """

    height, width = img_bgr.shape[:2]
    # Kiểm tra xem config có các tham số của bounding box hay không
    roi = config.get("roi")
    if roi is not None:
        # Kiểm tra roi có phải là một sequence có 4 giá trị hay không
        if not isinstance(roi, Sequence) or len(roi) != 4:
            raise ValueError("config['roi'] phải là một sequence có 4 giá trị: (x, y, w, h).")
        x, y, roi_w, roi_h = [int(v) for v in roi]
        x, y, roi_w, roi_h = _clip_roi(x, y, roi_w, roi_h, width, height)
        # Cắt ảnh bằng cơ chế Numpy Slicing: [start_y : end_y, start_x : end_x]
        return img_bgr[y : y + roi_h, x : x + roi_w]
    # Kiểm tra auto_roi có được kích hoạt hay không (mặc định là True)
    if not bool(config.get("auto_roi", True)):
        return img_bgr

    # 1. Phân tách nền bằng không gian màu HSV

    # BGR gộp chung ánh sáng và màu sắc => nhiễu bóng râm
    # HSV tách biệt sắc thái màu (Hue) ra khỏi độ bão hòa (Saturation) và độ sáng (Value)
    # => Tìm quả dưới các điều kiện ánh sáng khác nhau sẽ chính xác hơn
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Thiết lập ngưỡng tối thiểu cho độ bão hòa và độ sáng để nhận diện quả (mặc định lần lượt là 35 và 30)
    # TODO
    sat_thresh = int(config.get("auto_roi_sat_thresh", 35))
    val_thresh = int(config.get("auto_roi_val_thresh", 30))
    # hsv[:, :, 1] (Kênh Saturation) > sat_thresh: quả thường có độ bão hòa cao hơn nền
    # hsv[:, :, 2] (Kênh Value) > val_thresh: quả thường có độ sáng cao hơn nền hoặc bóng râm
    # True: pixel có khả năng thuộc về quả
    # False: pixel có khả năng thuộc về nền hoặc bóng râm
    # astype(np.uint8) * 255: chuyển đổi boolean mask thành ảnh nhị phân (True -> 255 hoặc False -> 0)
    mask = np.logical_and(hsv[:, :, 1] > sat_thresh, hsv[:, :, 2] > val_thresh).astype(np.uint8) * 255

    # 2. Khử nhiễu bằng Morphological Operations

    # Thiết lập kích thước Structuring Element (kernel) (mặc định là 5)
    # TODO
    kernel_size = int(config.get("auto_roi_morph_kernel", 5))
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

    # 3. Tìm đường viền và trích xuất bounding box

    # Trích xuất đường viền: RETR_EXTERNAL chỉ lấy đường viền ngoài cùng
    # Lưu trữ đường viền: CHAIN_APPROX_SIMPLE sẽ nén đường viền bằng cách loại bỏ các điểm dư thừa trên 1 đường thẳng
    # (chỉ lưu lại 4 điểm ở 4 góc)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Kiểm tra ảnh có tìm thấy đường viền nào hay không (không phải ảnh trắng hoặc đen toàn bộ)
    if not contours:
        return img_bgr
    
    # Tìm đường viền có diện tích lớn nhất (với bài toán "Single Object")
    # cv2.contourArea: tính diện tích của đường viền
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    # Thiết lập tỷ lệ diện tích đường viền tối thiểu để nhận diện là quả (mặc định là 0.05)
    # TODO
    min_area_ratio = float(config.get("auto_roi_min_area_ratio", 0.05))
    # Kiểm tra diện tích đường viền lớn nhất > diện tích đường viền tối thiểu (5% diện tích ảnh gốc) hay không
    if area < min_area_ratio * float(height * width):
        return img_bgr

    # Các tham số của bounding box có các cạnh // trục tọa độ nhỏ nhất của đường viền lớn nhất
    x, y, roi_w, roi_h = cv2.boundingRect(largest)
    # Thiết lập tỷ lệ padding bounding box sau khi tìm thấy (mặc định là 0.05)
    # Vì các mô hình học máy thường cần một chút bối cảnh (context) xung quanh đường viền vật thể
    # để trích xuất đặc trưng hình dáng chính xác
    # TODO
    pad_ratio = float(config.get("roi_padding_ratio", 0.05))
    pad_x = int(roi_w * pad_ratio)
    pad_y = int(roi_h * pad_ratio)
    # Cập nhật các tham số của bounding box sau khi padding
    x, y, roi_w, roi_h = _clip_roi(
        x - pad_x,
        y - pad_y,
        roi_w + (2 * pad_x), # 2 * pad_x: padding cả hai bên trái và phải
        roi_h + (2 * pad_y), # 2 * pad_y: padding cả hai bên trên và dưới
        width,
        height,
    )
    return img_bgr[y : y + roi_h, x : x + roi_w]


def _normalize_rgb(img_rgb: np.ndarray, normalize: bool) -> np.ndarray:
    """
    Chuẩn hóa kiểu dữ liệu các pixel của ảnh đầu vào về float32 và giá trị từ [0, 255] về [0, 1] (nếu được yêu cầu).

    Tham số:
        img_rgb (np.ndarray): Ma trận ảnh đầu vào (RGB).
        normalize (bool): Quyết định chuẩn hóa giá trị pixel về [0, 1] hay không.

    Trả về:
        np.ndarray: Ma trận ảnh với các pixel thuộc kiểu dữ liệu float32 và giá trị [0, 1] (nếu được yêu cầu).
    """
    rgb = img_rgb.astype(np.float32)
    if normalize:
        rgb = rgb / 255.0
    return rgb


def resize_and_normalize(img: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Hàm entry-point cho bước "2.1 Resize & Normalize" trong pipeline "2. Image Preprocessing".
    
    Tham số:
        img (np.ndarray): Ma trận ảnh đầu vào đọc từ source.
        config (Optional[Dict[str, Any]]): Cấu hình tùy biến pipeline. Hỗ trợ các key:
            - input_color_space (str): 'bgr' (mặc định), 'rgb', 'gray'.
            - target_size (Tuple[int, int]): Kích thước để resize (width, height) ma trận ảnh đầu vào, mặc định (256, 256).
            - normalize (bool): Quyết định chuẩn hóa giá trị pixel về [0, 1] hay không (mặc định True).
            - Và các key hỗ trợ bởi hàm _extract_roi_with_config().

    Trả về:
        np.ndarray: Ma trận ảnh với các pixel thuộc kiểu dữ liệu float32 sẵn sàng để trích xuất đặc trưng hình ảnh.

    Ngoại lệ:
        ValueError: Ảnh đầu vào không tồn tại hoặc không hợp lệ.
    """
    if img is None:
        raise ValueError("Ảnh đầu vào không tồn tại hoặc không hợp lệ.")

    cfg = config or {}

    # 1. Chuẩn hóa số lượng kênh và không gian màu của ma trận ảnh đầu vào
    src = _ensure_three_channels(img)
    input_color_space = str(cfg.get("input_color_space", "bgr")).lower()
    src_bgr = _to_bgr(src, input_color_space)

    # 2. Xác định và trích xuất ROI chứa quả từ ảnh gốc
    roi_img = _extract_roi_with_config(src_bgr, cfg)

    # 3. Resize về kích thước tiêu chuẩn cho mục "4. Feature Extraction"
    target_size = cfg.get("target_size", (256, 256))
    if not isinstance(target_size, Sequence) or len(target_size) != 2:
        raise ValueError("config['target_size'] phải là một tuple (width, height).")
    target_w, target_h = [int(v) for v in target_size]
    if target_w <= 0 or target_h <= 0:
        raise ValueError("Giá trị width và height của config['target_size'] phải > 0.")
    # Phép nội suy: INTER_AREA để tránh hiện tượng răng cưa xảy ra khi thu nhỏ ảnh
    resized = cv2.resize(roi_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 4. Chuyển sang không gian màu RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 5. Chuẩn hóa kiểu dữ liệu và giá trị pixel (mặc định là True)
    normalize = bool(cfg.get("normalize", True))
    return _normalize_rgb(rgb, normalize=normalize)
