"""FastAPI application for Caries Screening."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from web.backend.routers import analyze
from web.backend.services import inference

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Friendly display names for models
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "pano_caries_only_gpu2": "Panoramic",
    "pano_caries_only": "Avocado",
    "pano_caries_roboflow_v1": "Kiwi",
    "bitewing_caries_only": "Bitewing",
    "pano_dc1000_potato": "Potato",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    models = inference.find_models()
    if models:
        inference.load_model(models[0])
    yield


app = FastAPI(title="Caries Screening API", lifespan=lifespan)

origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(analyze.router)


@app.get("/api/models")
async def list_models():
    models = inference.find_models()
    result = []
    for m in models:
        raw_name = Path(m).parent.parent.name
        display = MODEL_DISPLAY_NAMES.get(raw_name, raw_name)
        result.append({"path": m, "name": display})
    return result
