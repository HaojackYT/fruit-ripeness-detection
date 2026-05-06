import tensorflow as tf
import numpy as np
import config


def predict_single_image(image_path):
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

    with open(config.CLASSES_SAVE_PATH, 'r', encoding='utf-8') as f:
        class_names = f.read().splitlines()

    img = tf.keras.utils.load_img(image_path, target_size=config.IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    print(f"Kết quả: {predicted_class} (Độ tự tin: {confidence:.2f}%)")


if __name__ == '__main__':
    # Test thử 1 ảnh ngẫu nhiên trong tập val
    # Nhớ thay đổi đường dẫn này trúng với ảnh thực tế trên máy bạn
    test_img = 'dataset/val/xoai_chin/image_1.jpg'
    predict_single_image(test_img)