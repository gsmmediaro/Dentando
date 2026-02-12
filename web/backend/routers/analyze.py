"""POST /api/analyze — image upload and inference."""

import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from web.backend.models import AnalysisResult, Detection
from web.backend.services import inference

router = APIRouter()
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/api/analyze", response_model=AnalysisResult)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    model_path: str = Form("auto"),
    conf_threshold: float = Form(0.25),
    modality: str = Form("Auto"),
    use_tooth_assignment: bool = Form(False),
    patient_name: str = Form(""),
):
    start = time.time()

    contents = await file.read()
    np_buf = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Auto-select model based on image dimensions
    if model_path == "auto":
        h, w = image.shape[:2]
        auto_model_path = inference.pick_model_for_image(h, w)
        if auto_model_path is None:
            raise HTTPException(status_code=400, detail="No trained models found")
        model_path = auto_model_path
        logger.info("Auto-selected model: %s", Path(model_path).parent.parent.name)

    model_name = Path(model_path).parent.parent.name

    result = inference.run_analysis(
        image_array=image,
        model_path=model_path,
        conf_threshold=conf_threshold,
        modality=modality,
        use_tooth_assignment=use_tooth_assignment,
    )

    # Save annotated image
    img_id = uuid.uuid4().hex[:12]
    img_filename = f"{img_id}.jpg"
    img_path = RESULTS_DIR / img_filename
    cv2.imwrite(str(img_path), result["annotated_image"])
    annotated_url = str(request.url_for("static", path=f"results/{img_filename}"))

    turnaround = round(time.time() - start, 2)

    detections = [
        Detection(
            class_name=d["class"],
            confidence=d["confidence"],
            bbox=list(d["bbox"]),
        )
        for d in result["detections"]
    ]

    return AnalysisResult(
        filename=file.filename or "unknown",
        suspicion_level=result["suspicion_level"],
        overall_confidence=result["overall_confidence"],
        detections=detections,
        annotated_image_url=annotated_url,
        modality=result["modality"],
        model_name=model_name,
        num_detections=result["num_detections"],
        turnaround_s=turnaround,
    )
