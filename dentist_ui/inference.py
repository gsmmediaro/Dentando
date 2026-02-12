"""Model loading, YOLO inference, and suspicion classification."""

import logging
from enum import Enum
from typing import List, Optional

import numpy as np
import streamlit as st

logger = logging.getLogger(__name__)


class SuspicionLevel(Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    REVIEW = "REVIEW"


def classify_suspicion(detections: List[dict]) -> SuspicionLevel:
    """Classify overall suspicion from a list of detections.

    Each detection dict must have a ``confidence`` key.
    """
    if not detections:
        return SuspicionLevel.LOW

    confs = [d["confidence"] for d in detections]
    max_conf = max(confs)
    n = len(confs)

    if all(c < 0.30 for c in confs):
        return SuspicionLevel.REVIEW
    if n >= 2 and max_conf >= 0.70:
        return SuspicionLevel.HIGH
    if n >= 1 and max_conf >= 0.40:
        return SuspicionLevel.MODERATE
    return SuspicionLevel.REVIEW


@st.cache_resource
def load_model(model_path: str):
    """Load a YOLO model, cached across Streamlit reruns."""
    from ultralytics import YOLO
    return YOLO(model_path)


def detect_modality(model_path: str) -> str:
    """Infer imaging modality from the model path string."""
    lower = model_path.lower()
    if "bitewing" in lower:
        return "Bitewing"
    if "pano" in lower or "dentex" in lower:
        return "Panoramic"
    return "Unknown"


def run_analysis(
    image_array: np.ndarray,
    model_path: str,
    conf_threshold: float = 0.25,
    modality: str = "Auto",
    use_tooth_assignment: bool = False,
) -> dict:
    """Run the full inference pipeline on a single image.

    Returns a dict with keys: detections, tooth_predictions, annotated_image,
    suspicion_level, overall_confidence, modality, num_detections.
    """
    model = load_model(model_path)
    results = model.predict(image_array, conf=conf_threshold, verbose=False)

    # Extract detections
    detections: List[dict] = []
    boxes = results[0].boxes
    names_map = results[0].names
    if boxes is not None and len(boxes) > 0:
        xyxy_list = boxes.xyxy.tolist()
        cls_ids = boxes.cls.tolist()
        confs = boxes.conf.tolist()
        for (x1, y1, x2, y2), class_id, score in zip(xyxy_list, cls_ids, confs):
            detections.append({
                "class": names_map.get(int(class_id), str(int(class_id))),
                "class_id": int(class_id),
                "confidence": round(float(score), 3),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
            })

    # Resolve modality
    resolved_modality = modality if modality != "Auto" else detect_modality(model_path)

    # Tooth-level predictions
    tooth_predictions = []
    if detections:
        try:
            from dental_tooth_caries_ai.tooth_level.assign_lesions_to_teeth import (
                make_direct_tooth_predictions,
            )
            class_names = list(names_map.values())
            tooth_predictions = make_direct_tooth_predictions(detections, class_names)

            if use_tooth_assignment and resolved_modality == "Bitewing":
                try:
                    from dental_tooth_caries_ai.tooth_level.tooth_proposals import (
                        propose_teeth_heuristic,
                    )
                    from dental_tooth_caries_ai.tooth_level.assign_lesions_to_teeth import (
                        assign_lesions_to_teeth,
                    )
                    teeth = propose_teeth_heuristic(image_array)
                    if teeth:
                        lesion_boxes = [d["bbox"] for d in detections]
                        lesion_classes = [d["class"] for d in detections]
                        lesion_confs = [d["confidence"] for d in detections]
                        tooth_predictions = assign_lesions_to_teeth(
                            teeth, lesion_boxes, lesion_classes, lesion_confs,
                        )
                except Exception as e:
                    logger.warning("Tooth assignment failed, using direct: %s", e)
        except Exception as e:
            logger.warning("Tooth prediction failed: %s", e)

    # Annotated image
    annotated_image = None
    if tooth_predictions:
        try:
            from dental_tooth_caries_ai.tooth_level.render_overlays import render_overlay
            annotated_image = render_overlay(image_array, tooth_predictions)
        except Exception:
            annotated_image = results[0].plot()
    else:
        annotated_image = results[0].plot()

    suspicion = classify_suspicion(detections)
    overall_conf = max((d["confidence"] for d in detections), default=0.0)

    return {
        "detections": detections,
        "tooth_predictions": tooth_predictions,
        "annotated_image": annotated_image,
        "suspicion_level": suspicion.value,
        "overall_confidence": round(overall_conf, 3),
        "modality": resolved_modality,
        "num_detections": len(detections),
    }
