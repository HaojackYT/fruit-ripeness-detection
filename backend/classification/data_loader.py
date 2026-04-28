import tensorflow as tf
import os
import config

# Định nghĩa từ điển để chuyển chữ thành số (AI chỉ hiểu số)
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


def parse_image(file_path, fruit_label, ripe_label):
    # Đọc file ảnh từ ổ cứng
    img = tf.io.read_file(file_path)
    # Giải mã ảnh JPEG/PNG
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize về 224x224
    img = tf.image.resize(img, config.IMG_SIZE)

    # Quan trọng nhất: Trả về 1 ảnh đi kèm với 1 tuple chứa 2 nhãn
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