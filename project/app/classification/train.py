import os
import sys

# Đưa thư mục project vào PYTHONPATH để Python hiểu được "from app..."
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

import tensorflow as tf
import config
from data_loader import get_datasets, get_augmentation
from model import build_efficientnet_model


def train():
    # 1. Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    # 2. Nạp dữ liệu đa nhãn
    train_dataset, val_dataset = get_datasets()
    augmentation_layer = get_augmentation()

    print(f"Hệ thống huấn luyện: {config.FRUIT_CLASSES} và {config.RIPE_CLASSES}")

    # 3. Xây dựng mô hình 2 nhánh
    # Truyền vào số lượng lớp của cả 2 nhánh từ config
    model = build_efficientnet_model(
        input_shape=(*config.IMG_SIZE, 3),
        num_fruit_classes=config.NUM_FRUIT_CLASSES,
        num_ripe_classes=config.NUM_RIPE_CLASSES,
        augmentation_layer=augmentation_layer
    )

    # 4. Biên dịch với 2 hàm Loss riêng biệt
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'fruit_out': 'sparse_categorical_crossentropy',
            'ripe_out': 'sparse_categorical_crossentropy'
        },
        metrics={
            'fruit_out': 'accuracy',
            'ripe_out': 'accuracy'
        }
    )

    # 5. Cấu hình CHỈ LƯU 1 MODEL TỐT NHẤT
    # Đặt tên file cố định để Keras tự động ghi đè lên file cũ khi có model tốt hơn
    # Cấu hình lưu Model xịn nhất
    checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    # VŨ KHÍ MỚI: Tự động dừng khi hết tiến triển
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Vẫn theo dõi lỗi trên tập Validation
        patience=5,  # Nếu sau 5 Epochs mà val_loss không giảm thì dừng luôn
        restore_best_weights=True  # Tự động lùi về phiên bản có trọng số tốt nhất
    )

    print("Bắt đầu quá trình huấn luyện Multi-task...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,  # Cứ set 50 thoải mái, nó sẽ tự dừng sớm
        callbacks=[checkpoint_callback, early_stopping_callback]  # Gọi cả 2 callback vào
    )

    print(f"Huấn luyện hoàn tất! Model xuất sắc nhất đã được lưu tại: {checkpoint_path}")


if __name__ == '__main__':
    train()