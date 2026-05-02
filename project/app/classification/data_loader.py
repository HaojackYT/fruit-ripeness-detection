import sys
import os

# Đưa thư mục project vào PYTHONPATH để Python hiểu được "from app..."
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

import cv2
import numpy as np
import tensorflow as tf
import config
from app.image_preprocessing.pipeline import preprocess_image, _hsv_to_bgr_u8

# Định nghĩa từ điển để chuyển chữ thành số
FRUIT_LABELS = {'apple': 0, 'mango': 1, 'orange': 2}
RIPE_LABELS = {'ripe': 0, 'unripe': 1}


def load_paths_and_labels(base_dir):
    file_paths, fruit_targets, ripe_targets = [], [], []

    for fruit in os.listdir(base_dir):
        if fruit not in FRUIT_LABELS: continue
        fruit_path = os.path.join(base_dir, fruit)

        for ripeness in os.listdir(fruit_path):
            if ripeness not in RIPE_LABELS: continue
            ripe_path = os.path.join(fruit_path, ripeness)

            for file in os.listdir(ripe_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_paths.append(os.path.join(ripe_path, file))
                    fruit_targets.append(FRUIT_LABELS[fruit])
                    ripe_targets.append(RIPE_LABELS[ripeness])

    return file_paths, fruit_targets, ripe_targets


def python_preprocess(file_path_tensor):
    file_path = file_path_tensor.numpy().decode('utf-8')
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)

    preprocessed_hsv = preprocess_image(
        img,
        config={
            "input_color_space": "bgr",
            "target_size": config.IMG_SIZE,
            "normalize": False
        }
    )

    processed_bgr = _hsv_to_bgr_u8(preprocessed_hsv)
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

    return processed_rgb.astype(np.float32)

def parse_image(file_path, fruit_label, ripe_label):
    img = tf.py_function(func=python_preprocess, inp=[file_path], Tout=tf.float32)
    img.set_shape([*config.IMG_SIZE, 3])
    return img, (fruit_label, ripe_label)


def get_datasets():
    # --- Xử lý tập Train ---
    train_paths, train_fruits, train_ripes = load_paths_and_labels(config.DATA_DIR_TRAIN)
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_fruits, train_ripes))
    # Dùng num_parallel_calls để load ảnh đa luồng cho nhanh
    train_ds = train_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(1000).batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # --- Xử lý tập Validation ---
    val_paths, val_fruits, val_ripes = load_paths_and_labels(config.DATA_DIR_VAL)
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_fruits, val_ripes))
    val_ds = val_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def get_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1),
    ])