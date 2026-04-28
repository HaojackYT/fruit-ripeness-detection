import os

IMG_SIZE = (224, 224)   # kích cỡ ảnh chuẩn cho EfficientNetB0
BATCH_SIZE = 32         # số lượng ảnh trong mỗi lần đưa vào mô hình để huấn luyện
EPOCHS = 15             # số vòng lặp huấn luyện trên toàn bộ tập dữ liệu

FRUIT_CLASSES = ['apple', 'mango', 'orange'] # 3 loại trái cây chính trong dataset, sẽ được dùng để tạo nhãn cho bài toán đa nhãn
RIPE_CLASSES = ['ripe', 'unripe']            # 2 trạng thái chín của trái cây, cũng sẽ được dùng để tạo nhãn cho bài toán đa nhãn

NUM_FRUIT_CLASSES = len(FRUIT_CLASSES)      # Sẽ bằng 3
NUM_RIPE_CLASSES = len(RIPE_CLASSES)        # Sẽ bằng 2

DATA_DIR_TRAIN = 'backend/classification/dataset/train'            # Đường dẫn đến thư mục chứa ảnh huấn luyện.
DATA_DIR_VAL = 'backend/classification/dataset/val'                # Đường dẫn đến thư mục chứa ảnh validation.

MODEL_SAVE_DIR = 'backend/classification/saved_model'