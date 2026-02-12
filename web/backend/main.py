"""FastAPI application for Caries Screening."""

import logging
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

LOGGER = logging.getLogger(__name__)

# Friendly display names for models
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "pano_caries_only_gpu2": "Panoramic",
    "pano_gpu2": "Panoramic",
    "bitewing_caries_only": "Bitewing",
    "bitewing": "Bitewing",
    "pano_caries_roboflow_v1": "Kiwi",
    "pano_dc1000_potato": "Kiwi",
    "potato": "Kiwi",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    models = inference.find_models()
    if models:
        inference.load_model(models[0])
    yield


app = FastAPI(title="Caries Screening API", lifespan=lifespan)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


origins = _split_csv(
    os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
)

for key in ("FRONTEND_ORIGIN", "FRONTEND_URL"):
    value = (os.getenv(key) or "").strip().rstrip("/")
    if value and value not in origins:
        origins.append(value)

origin_regex = os.getenv("CORS_ORIGIN_REGEX")
if origin_regex is None:
    origin_regex = r"^https://([a-zA-Z0-9-]+\.)?vercel\.app$"
else:
    origin_regex = origin_regex.strip() or None

LOGGER.info("CORS config loaded origins=%s origin_regex=%s", origins, origin_regex)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=origin_regex if origin_regex else None,
    allow_credentials=True,
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
