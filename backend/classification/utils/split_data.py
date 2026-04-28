import os
import shutil
import random


def split_nested_dataset(source_dir, dest_dir, split_ratio=0.8):
    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'val')

    # 1. Duyệt qua tầng 1: Loại quả (apple, mango, orange)
    fruits = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for fruit in fruits:
        fruit_path = os.path.join(source_dir, fruit)

        # 2. Duyệt qua tầng 2: Độ chín (unripe, ripe)
        ripeness_levels = [d for d in os.listdir(fruit_path) if os.path.isdir(os.path.join(fruit_path, d))]

        for ripe_level in ripeness_levels:
            ripe_path = os.path.join(fruit_path, ripe_level)

            # Tạo sẵn thư mục đích: vd dest/train/orange/unripe
            target_train_path = os.path.join(train_dir, fruit, ripe_level)
            target_val_path = os.path.join(val_dir, fruit, ripe_level)
            os.makedirs(target_train_path, exist_ok=True)
            os.makedirs(target_val_path, exist_ok=True)

            # Lấy tất cả ảnh jpg/png
            images = [f for f in os.listdir(ripe_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(images)

            # Cắt điểm 80/20
            split_point = int(len(images) * split_ratio)
            train_images = images[:split_point]
            val_images = images[split_point:]

            # Copy ảnh
            for img in train_images:
                shutil.copy(os.path.join(ripe_path, img), os.path.join(target_train_path, img))
            for img in val_images:
                shutil.copy(os.path.join(ripe_path, img), os.path.join(target_val_path, img))

            print(f"[{fruit}/{ripe_level}]: {len(train_images)} train, {len(val_images)} val.")


if __name__ == '__main__':
    # TRỎ VÀO THƯ MỤC DATASET CỦA BẠN (Đã dọn sạch file .webp)
    SOURCE_DIRECTORY = 'dataset_raw'  # Thư mục gốc chứa ảnh
    DESTINATION_DIRECTORY = 'dataset'  # Thư mục đích sẽ dùng để train

    split_nested_dataset(SOURCE_DIRECTORY, DESTINATION_DIRECTORY)
    print("Đã chia thành công 80% Train và 20% Val!")