# Fruit Ripeness Detection

Cấu trúc thư mục đã được tạo.

Chay thu server (PowerShell):

```powershell
cd "c:/Users/Tuan Phat/Downloads/fruit-ripeness-detection/project"
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Mở trình duyệt: http://127.0.0.1:8000

## API hien tai

- `GET /api/health`
	Kiem tra trang thai service cho moi truong deploy/monitoring.

- `GET /api/model/info`
	Tra ve metadata mo hinh: classifier SVM, feature set va class labels.

- `POST /api/predict`
	- Du doan 1 anh (single object).
	- `form-data`: `file` (image)
	- Tra ve them truong `segmentation` (neu thanh cong):
		- `overlay_image`: data URL de hien thi len frontend
		- `mask_ratio`: ty le vung doi tuong
		- `bbox`: khung bao lon nhat (x,y,w,h)

