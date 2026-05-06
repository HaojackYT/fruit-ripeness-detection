from __future__ import annotations

import os
import uuid
from typing import Any

import cv2
import numpy as np
import tensorflow as tf

from app.image_preprocessing.pipeline import preprocess_image, _hsv_to_bgr_u8

MODEL_NAME = "EfficientNetB0 (CNN)"
MODEL_VERSION = "1.0.0"
FEATURE_SET = "deep_features"
PIPELINE_SCOPE = "single_object_realistic_condition"
SUPPORTED_FRUITS = ["apple", "mango", "orange"]

# Tải mô hình toàn cục để không phải tải lại mỗi lần gọi API
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'classification', 'saved_model', 'best_model.h5')

try:
    print(f"Đang tải mô hình CNN từ: {model_path}")
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    model = None
    print(f"LỖI: Không thể tải mô hình: {e}")


def predict_image(image_bytes: bytes) -> dict[str, Any]:
    """Predict fruit type and ripeness for a single image using CNN."""
    prediction_id = uuid.uuid4().hex[:12]
    
    if model is None:
        raise RuntimeError("Mô hình CNN chưa được tải thành công.")

    # 1. Chuyển đổi bytes thành numpy array và đọc ảnh
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Dữ liệu ảnh không hợp lệ.")

    # 2. Chuyển sang RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 3. Tiền xử lý (giống với predict.py trong classification)
    processed_hsv = preprocess_image(
        img_rgb,
        config={
            "input_color_space": "rgb",
            "target_size": (224, 224),
            "normalize": False
        }
    )
    processed_bgr = _hsv_to_bgr_u8(processed_hsv)
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

    # 4. Mở rộng chiều (batch dimension)
    img_array = tf.expand_dims(processed_rgb, 0)

    # 5. Dự đoán
    predictions = model.predict(img_array)

    fruit_pred = predictions[0]
    ripe_pred = predictions[1]

    # Nhãn cấu hình
    FRUIT_CLASSES = ['apple', 'mango', 'orange']
    RIPE_CLASSES = ['ripe', 'unripe']

    predicted_fruit = FRUIT_CLASSES[np.argmax(fruit_pred[0])]
    fruit_confidence = float(np.max(fruit_pred[0]))

    predicted_ripe = RIPE_CLASSES[np.argmax(ripe_pred[0])]
    ripe_confidence = float(np.max(ripe_pred[0]))

    # Độ tin cậy trung bình để tương thích với frontend hiện tại
    confidence = (fruit_confidence + ripe_confidence) / 2.0

    result_label = f"{predicted_fruit.title()} {'Ripe' if predicted_ripe == 'ripe' else 'Unripe'}"

    return {
        "prediction_id": prediction_id,
        "fruit_type": predicted_fruit,
        "ripeness": predicted_ripe,
        "ripeness_vi": "chín" if predicted_ripe == "ripe" else "xanh",
        "result": result_label,
        "confidence": round(confidence, 4),
        "fruit_confidence": round(fruit_confidence, 4),
        "ripeness_confidence": round(ripe_confidence, 4),
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "pipeline_scope": PIPELINE_SCOPE,
    }


def get_model_info() -> dict[str, Any]:
    """Return model metadata for frontend and monitoring."""
    return {
        "task": "fruit_type_and_ripeness_classification",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "ml_paradigm": "deep_learning",
        "classifier": "cnn_efficientnetb0",
        "feature_set": FEATURE_SET,
        "classes": {
            "fruit_type": SUPPORTED_FRUITS,
            "ripeness": ["unripe", "ripe"],
        },
        "pipeline_scope": PIPELINE_SCOPE,
    }
