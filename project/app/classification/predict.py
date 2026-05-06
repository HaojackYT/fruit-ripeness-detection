import os
import sys

# Đưa thư mục project vào PYTHONPATH để Python hiểu được "from app..."
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

import cv2
import tensorflow as tf
import numpy as np
import config

from app.image_preprocessing.pipeline import preprocess_image, _hsv_to_bgr_u8


def predict_single_image(image_path):
    print("Đang tải mô hình...")
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

    # 1. Đọc ảnh thô
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"LỖI: Không thể đọc ảnh từ đường dẫn: {image_path}")
        return

    # 2. CHUYỂN SANG RGB NGAY LẬP TỨC (Sửa lỗi màu sắc)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 3. Đưa ảnh RGB chuẩn vào Pipeline Tiền xử lý
    processed_hsv = preprocess_image(
        img_rgb,
        config={
            "input_color_space": "rgb",
            "target_size": config.IMG_SIZE,
            "normalize": False
        }
    )
    processed_bgr = _hsv_to_bgr_u8(processed_hsv)
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

    # 5. Mở rộng chiều (batch dimension)
    img_array = tf.expand_dims(processed_rgb, 0)

    # Model returns [fruit_output, ripe_output]
    predictions = model.predict(img_array)

    fruit_pred = predictions[0]
    ripe_pred = predictions[1]

    predicted_fruit = config.FRUIT_CLASSES[np.argmax(fruit_pred[0])]
    fruit_confidence = 100 * np.max(fruit_pred[0])

    predicted_ripe = config.RIPE_CLASSES[np.argmax(ripe_pred[0])]
    ripe_confidence = 100 * np.max(ripe_pred[0])

    print(f"Loại quả: {predicted_fruit} (Độ tự tin: {fruit_confidence:.2f}%)")
    print(f"Độ chín: {predicted_ripe} (Độ tự tin: {ripe_confidence:.2f}%)")


if __name__ == '__main__':
    test_img = config.IMAGE_PATH
    predict_single_image(test_img)