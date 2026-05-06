import tensorflow as tf
import os
import config
from data_loader import get_datasets, get_augmentation
from model import build_efficientnet_model


def train():
    # 1. Đảm bảo thư mục lưu trữ tồn tại [cite: 524]
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    # 2. Nạp dữ liệu đa nhãn [cite: 532]
    train_dataset, val_dataset = get_datasets()
    augmentation_layer = get_augmentation()

    print(f"Hệ thống huấn luyện: {config.FRUIT_CLASSES} và {config.RIPE_CLASSES}")

    # 3. Xây dựng mô hình 2 nhánh [cite: 515, 544]
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

    # 5. Cấu hình lưu nhiều phiên bản (Checkpoint) [cite: 526, 527]
    checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, "model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,  # Chỉ lưu những phiên bản tốt nhất để tiết kiệm bộ nhớ [cite: 528, 529]
        monitor='val_loss',
        verbose=1
    )

    print("Bắt đầu quá trình huấn luyện Multi-task...")
    # Khi fit, Keras sẽ tự khớp tuple nhãn từ data_loader vào 2 nhánh đầu ra [cite: 546, 551]
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        callbacks=[checkpoint_callback]
    )

    print(f"Huấn luyện hoàn tất! Các phiên bản model nằm trong: {config.MODEL_SAVE_DIR}")


if __name__ == '__main__':
    train()