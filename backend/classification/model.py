import tensorflow as tf
import config

def get_datasets():
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        config.DATA_DIR_TRAIN,
        shuffle=True,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE
    )
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        config.DATA_DIR_VAL,
        shuffle=True,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE
    )
    return train_dataset, val_dataset

def get_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1), # Thêm zoom nhẹ để ảnh đa dạng hơn
    ])