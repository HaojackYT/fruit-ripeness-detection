from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import config

app = FastAPI(title="Hệ thống phân loại Trái cây (API)")

print("Đang khởi động hệ thống AI...")
model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

with open(config.CLASSES_SAVE_PATH, 'r', encoding='utf-8') as f:
    CLASS_NAMES = f.read().splitlines()


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')

        img = img.resize(config.IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        predicted_class = CLASS_NAMES[np.argmax(score)]
        confidence = float(np.max(score)) * 100

        return {
            "status": "success",
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": f"{confidence:.2f}%"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Chạy server bằng lệnh trong terminal:
# uvicorn api:app --reload