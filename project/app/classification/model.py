import tensorflow as tf

def build_efficientnet_model(input_shape, num_fruit_classes, num_ripe_classes, augmentation_layer):
    # 1. Khởi tạo mô hình cơ sở EfficientNetB0 (Bỏ lớp phân loại ở đuôi)
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Đóng băng các trọng số của mô hình cơ sở (Transfer Learning)
    base_model.trainable = True

    for layer in base_model.layers[:100]:
        layer.trainable = False

    # 2. Xây dựng đầu vào và đi qua các lớp chung
    inputs = tf.keras.Input(shape=input_shape)

    # Áp dụng Tăng cường dữ liệu (Augmentation)
    x = augmentation_layer(inputs)

    # Đi qua EfficientNet (training=False để giữ nguyên BatchNormalization)
    x = base_model(x, training=False)

    # Nén các đặc trưng 2D thành một vector 1 chiều
    shared_features = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Thêm Dropout để giảm thiểu học vẹt
    shared_features = tf.keras.layers.Dropout(0.2)(shared_features)

    # ---------------------------------------------------------
    # 3. NHÁNH 1: Dự đoán Loại trái cây (Xoài, Táo, Cam)
    # ---------------------------------------------------------
    # Cho nhánh này tự học thêm một chút đặc trưng riêng qua 1 lớp Dense
    fruit_branch = tf.keras.layers.Dense(128, activation='relu')(shared_features)
    # Đầu ra 1: Số lượng nơ-ron bằng số loại quả (3) [cite: 259]
    fruit_output = tf.keras.layers.Dense(num_fruit_classes, activation='softmax', name='fruit_out')(fruit_branch)

    # ---------------------------------------------------------
    # 4. NHÁNH 2: Dự đoán Độ chín (Xanh, Chín)
    # ---------------------------------------------------------
    # Cho nhánh này tự học thêm một chút đặc trưng riêng qua 1 lớp Dense
    ripe_branch = tf.keras.layers.Dense(128, activation='relu')(shared_features)
    # Đầu ra 2: Số lượng nơ-ron bằng số cấp độ chín (2) [cite: 260]
    ripe_output = tf.keras.layers.Dense(num_ripe_classes, activation='softmax', name='ripe_out')(ripe_branch)

    # ---------------------------------------------------------
    # 5. ĐÓNG GÓI MÔ HÌNH
    # ---------------------------------------------------------
    # Gộp 2 đầu ra thành một danh sách (list) [cite: 493]
    model = tf.keras.Model(inputs=inputs, outputs=[fruit_output, ripe_output])

    return model