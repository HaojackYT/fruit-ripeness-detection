import os
import shutil
import random

def split_nested_dataset_inplace(dataset_dir, split_ratio=0.8):
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')

    # 1. Duyệt qua tầng 1: Lấy danh sách quả (BẮT BUỘC bỏ qua thư mục 'train' và 'val')
    fruits = [d for d in os.listdir(dataset_dir)
              if os.path.isdir(os.path.join(dataset_dir, d)) and d not in ['train', 'val']]

    for fruit in fruits:
        fruit_path = os.path.join(dataset_dir, fruit)

        # 2. Duyệt qua tầng 2: Độ chín (unripe, ripe)
        ripeness_levels = [d for d in os.listdir(fruit_path) if os.path.isdir(os.path.join(fruit_path, d))]

        for ripe_level in ripeness_levels:
            ripe_path = os.path.join(fruit_path, ripe_level)

            # Tạo sẵn thư mục đích: vd dataset/train/orange/unripe
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

            # DI CHUYỂN (Move) ảnh thay vì Copy để dọn dẹp thư mục cũ
            for img in train_images:
                shutil.move(os.path.join(ripe_path, img), os.path.join(target_train_path, img))
            for img in val_images:
                shutil.move(os.path.join(ripe_path, img), os.path.join(target_val_path, img))

            print(f"[{fruit}/{ripe_level}]: Đã di chuyển {len(train_images)} train, {len(val_images)} val.")

        # 3. Dọn dẹp: Sau khi move hết ảnh, thư mục quả cũ (vd: dataset/orange) sẽ rỗng, ta xóa nó đi
        try:
            shutil.rmtree(fruit_path)
        except Exception as e:
            print(f"Không thể xóa thư mục cũ {fruit_path}: {e}")

if __name__ == '__main__':
    # 1. Lấy đường dẫn tuyệt đối của thư mục chứa file code này (utils)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Đi ngược ra 1 cấp, trỏ thẳng vào thư mục 'dataset'
    DATASET_DIR = os.path.normpath(os.path.join(current_dir, '..', 'dataset'))

    print(f"Đang tiến hành chia dữ liệu trực tiếp trong: {DATASET_DIR}")
    split_nested_dataset_inplace(DATASET_DIR)
    print("Hoàn tất! Dữ liệu đã được chia gọn gàng vào 2 thư mục 'train' và 'val'.")