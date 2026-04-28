import tensorflow as tf
import os
import config
from data_loader import get_datasets, get_augmentation
from model import build_efficientnet_model


def train():
    # Tạo thư mục lưu model nếu chưa có
    os.makedirs('saved_models', exist_ok=True)

    train_dataset, val_dataset = get_datasets()
    augmentation_layer = get_augmentation()
    class_names = train_dataset.class_names
    print(f"Hệ thống sẽ nhận diện {len(class_names)} lớp: {class_names}")

    # Xây dựng mô hình
    model = build_efficientnet_model((*config.IMG_SIZE, 3), config.NUM_CLASSES, augmentation_layer)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Bắt đầu quá trình huấn luyện...")
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=config.EPOCHS)

    # Lưu mô hình
    model.save(config.MODEL_SAVE_PATH)
    print(f"Đã lưu mô hình tại: {config.MODEL_SAVE_PATH}")

    # Lưu tên class để API sử dụng
    with open(config.CLASSES_SAVE_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names))


if __name__ == '__main__':
    train()