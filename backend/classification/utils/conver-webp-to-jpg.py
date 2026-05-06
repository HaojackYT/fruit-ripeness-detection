import os
from PIL import Image

def convert_webp_to_jpg(directory):
    count = 0
    # Duyệt qua tất cả các thư mục con và file bên trong
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Nếu phát hiện file đuôi .webp
            if file.lower().endswith(".webp"):
                webp_path = os.path.join(root, file)
                # Đổi đuôi tên file thành .jpg
                jpg_path = os.path.join(root, file.rsplit('.', 1)[0] + '.jpg')

                try:
                    # Mở file webp và chuyển đổi hệ màu sang RGB chuẩn của jpg
                    im = Image.open(webp_path).convert("RGB")
                    im.save(jpg_path, "jpeg")

                    # Tùy chọn: Xóa file .webp cũ đi để tiết kiệm dung lượng
                    os.remove(webp_path)
                    count += 1
                except Exception as e:
                    print(f"Lỗi khi xử lý file {file}: {e}")

    print(f"Đã chuyển đổi thành công {count} file .webp sang .jpg!")


# --- PHẦN ĐƯỜNG DẪN ĐÃ ĐƯỢC SỬA LẠI ---

# 1. Lấy đường dẫn tuyệt đối của thư mục 'utils' (nơi chứa file code này)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Đi ngược ra 1 cấp (để về thư mục 'classification'), sau đó nối với 'dataset'
THU_MUC_ANH = os.path.join(current_dir, '..', 'dataset')

# 3. Chuẩn hóa đường dẫn (tự động xóa bỏ các dấu '..' dư thừa để đường dẫn đẹp hơn)
THU_MUC_ANH = os.path.normpath(THU_MUC_ANH)

print(f"Đang quét tìm file .webp tại: {THU_MUC_ANH}")
convert_webp_to_jpg(THU_MUC_ANH)