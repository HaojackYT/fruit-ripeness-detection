import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import get_datasets


def evaluate_model():
    # 1. Tải mô hình xịn nhất đã lưu
    print("Đang tải mô hình...")
    import config
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

    # 2. Lấy tập dữ liệu Validation (không dùng shuffle để giữ đúng thứ tự)
    _, val_ds = get_datasets()

    print("Đang chạy dự đoán trên tập Validation (vui lòng đợi)...")

    y_true_fruit = []
    y_true_ripe = []
    y_pred_fruit = []
    y_pred_ripe = []

    # Duyệt qua tập Val để lấy nhãn thật và nhãn dự đoán
    for images, labels in val_ds:
        # labels là tuple (fruit_labels, ripe_labels) từ data_loader
        fruit_labels, ripe_labels = labels

        # Dự đoán
        preds = model.predict(images, verbose=0)

        # Lưu nhãn thật
        y_true_fruit.extend(fruit_labels.numpy())
        y_true_ripe.extend(ripe_labels.numpy())

        # Lưu nhãn dự đoán (lấy vị trí có xác suất cao nhất)
        y_pred_fruit.extend(np.argmax(preds[0], axis=1))
        y_pred_ripe.extend(np.argmax(preds[1], axis=1))

    # 3. Xuất bảng báo cáo cho nhánh Loại Quả (Fruit)
    print("\n" + "=" * 20 + " BÁO CÁO NHÁNH: LOẠI QUẢ " + "=" * 20)
    print(classification_report(
        y_true_fruit,
        y_pred_fruit,
        target_names=config.FRUIT_CLASSES,
        digits=4
    ))

    # 4. Xuất bảng báo cáo cho nhánh Độ Chín (Ripeness)
    print("\n" + "=" * 20 + " BÁO CÁO NHÁNH: ĐỘ CHÍN " + "=" * 20)
    print(classification_report(
        y_true_ripe,
        y_pred_ripe,
        target_names=config.RIPE_CLASSES,
        digits=4
    ))


if __name__ == '__main__':
    evaluate_model()