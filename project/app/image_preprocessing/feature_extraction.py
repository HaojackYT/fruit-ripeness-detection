import cv2
import numpy as np
from skimage.feature import local_binary_pattern

class FruitFeatureExtractor:
    def __init__(self, config=None):
        self.config = config or {}
        # Thông số cho LBP (Texture)
        self.lbp_radius = 3
        self.lbp_n_points = 8 * self.lbp_radius

    def extract_color_features(self, hsv_img):
        """Trích xuất giá trị trung bình và độ lệch chuẩn của các kênh H, S, V"""
        features = []
        for i in range(3):  # H, S, V channels
            channel = hsv_img[:, :, i]
            features.append(np.mean(channel))
            features.append(np.std(channel))
        return features

    def extract_texture_features(self, hsv_img):
        """Trích xuất đặc trưng kết cấu bằng Local Binary Pattern (LBP)"""
        # Sử dụng kênh V (Value) để tính toán kết cấu
        gray = (hsv_img[:, :, 2] * 255).astype(np.uint8)
        lbp = local_binary_pattern(gray, self.lbp_n_points, self.lbp_radius, method="uniform")
        
        # Tạo histogram từ LBP làm vector đặc trưng
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.lbp_n_points + 3), range=(0, self.lbp_n_points + 2))
        
        # Chuẩn hóa histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist.tolist()

    def extract_shape_features(self, hsv_img):
        """Trích xuất các đặc trưng hình dạng (diện tích, chu vi, độ tròn)"""
        # Tạo mask từ kênh Saturation và Value (tương tự bước preprocess bạn đã làm)
        mask = np.logical_and(hsv_img[:,:,1] > 0.12, hsv_img[:,:,2] > 0.12).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return [0.0, 0.0, 0.0] # Mặc định nếu không tìm thấy đối tượng

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Độ tròn (Compactness/Circularity): 4*pi*A / P^2
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        
        return [area / (hsv_img.shape[0] * hsv_img.shape[1]), perimeter, circularity]

    def get_all_features(self, hsv_img):
        """Hàm tổng hợp tất cả các đặc trưng thành 1 vector duy nhất"""
        color = self.extract_color_features(hsv_img)
        texture = self.extract_texture_features(hsv_img)
        shape = self.extract_shape_features(hsv_img)
        
        # Kết hợp thành một vector đặc trưng phẳng (Flatten vector)
        return np.array(color + texture + shape, dtype=np.float32)