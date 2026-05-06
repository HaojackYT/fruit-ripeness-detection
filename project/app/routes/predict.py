from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.services.model_service import (
    get_model_info,
    predict_image,
)

router = APIRouter()

MAX_IMAGE_SIZE_BYTES = 8 * 1024 * 1024


def _validate_image(file: UploadFile, content: bytes) -> None:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large. Max bensize is {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)} MB.",
        )


@router.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "service": "fruit-ripeness-detection-api",
            "scope": "single_object_realistic_condition",
        }
    )


@router.get("/model/info")
async def model_info() -> JSONResponse:
    return JSONResponse(get_model_info())

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    _validate_image(file, contents)

    prediction = predict_image(contents)
    prediction["filename"] = file.filename or "uploaded_image"

    # Keep result/confidence for existing frontend while adding richer metadata.
    return JSONResponse(prediction)
