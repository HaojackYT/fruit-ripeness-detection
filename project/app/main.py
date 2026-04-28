from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routes.predict import router as predict_router

app = FastAPI(title="Fruit Ripeness Detection")

# Mount static files and templates (run from project root)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Include API routes
app.include_router(predict_router, prefix="/api")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/result")
async def result_page(request: Request):
    return templates.TemplateResponse(request=request, name="result.html")
