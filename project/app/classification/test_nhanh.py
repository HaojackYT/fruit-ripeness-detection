import tensorflow as tf
import numpy as np
import config


def test_raw_model(image_path):
    print("Đang tải mô hình...")
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

    # 1. Đọc ảnh Y HỆT LÚC TRAIN (dùng thẳng tf.io)
    img_raw = tf.io.read_file(image_path)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=3)

    # 2. Resize Y HỆT LÚC TRAIN
    img_resized = tf.image.resize(img_tensor, config.IMG_SIZE)

    # 3. Mở rộng chiều và đưa vào dự đoán
    img_array = tf.expand_dims(img_resized, 0)
    predictions = model.predict(img_array)

    # Lấy Raw Output (Chưa qua tên nhãn)
    fruit_idx = np.argmax(predictions[0][0])
    ripe_idx = np.argmax(predictions[1][0])

    print("\n--- KẾT QUẢ RAW TỪ AI ---")
    print(f"Chỉ số loại quả: {fruit_idx} (Theo FRUIT_LABELS lúc train)")
    print(f"Chỉ số độ chín: {ripe_idx} (0 = Ripe, 1 = Unripe)")

    print("\n--- KẾT QUẢ SAU KHI MAP NHÃN ---")
    print(f"Tên quả: {config.FRUIT_CLASSES[fruit_idx]}")
    print(f"Tên độ chín: {config.RIPE_CLASSES[ripe_idx]}")


if __name__ == '__main__':
    # Bỏ đường dẫn ảnh xoài xanh của bạn vào đây
    test_img = '/home/hoang/Projects/BT_python/btl_computer_vision/backend/classification/dataset/val/mango/unripe/mango_unripe_054.jpg'
    test_raw_model(test_img)