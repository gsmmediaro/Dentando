#!/usr/bin/env python3
"""
Render tooth-level prediction overlays on dental images.

Draws bounding boxes around affected teeth with diagnosis labels and
confidence scores.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from .tooth_instance import ImagePredictions, ToothInstance, ToothPrediction

log = logging.getLogger(__name__)

# Color palette per diagnosis (BGR for OpenCV)
DIAGNOSIS_COLORS = {
    "caries":            (0, 165, 255),    # Orange
    "Caries":            (0, 165, 255),
    "deep_caries":       (0, 0, 220),      # Red
    "Deep Caries":       (0, 0, 220),
    "deep caries":       (0, 0, 220),
    "periapical_lesion": (255, 0, 128),    # Purple
    "Periapical Lesion": (255, 0, 128),
    "impacted":          (255, 200, 0),    # Cyan-ish
    "Impacted":          (255, 200, 0),
}
DEFAULT_COLOR = (0, 255, 0)  # Green
TOOTH_BOX_COLOR = (200, 200, 200)  # Light gray for unaffected teeth


def render_overlay(
    image: np.ndarray,
    predictions: List[ToothPrediction],
    all_teeth: Optional[List[ToothInstance]] = None,
    show_tooth_boxes: bool = True,
    show_lesion_boxes: bool = False,
    line_thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw tooth-level results on the image.

    Args:
        image: Input image (BGR numpy array). Will be copied, not modified.
        predictions: List of ToothPrediction with diagnosis labels.
        all_teeth: Optional list of all tooth instances (for drawing unaffected).
        show_tooth_boxes: Draw thin boxes around all detected teeth.
        show_lesion_boxes: Also draw the original lesion box (for mapped preds).
        line_thickness: Box line thickness.
        font_scale: Text font scale.

    Returns:
        Annotated image as BGR numpy array.
    """
    import cv2

    canvas = image.copy()
    h, w = canvas.shape[:2]

    # Draw all tooth boxes (subtle)
    if show_tooth_boxes and all_teeth:
        for tooth in all_teeth:
            x1, y1, x2, y2 = _int_box(tooth.bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), TOOTH_BOX_COLOR, 1)

    # Draw predictions
    for pred in predictions:
        color = DIAGNOSIS_COLORS.get(pred.diagnosis, DEFAULT_COLOR)
        x1, y1, x2, y2 = _int_box(pred.tooth.bbox)

        # Draw tooth box (bold)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_thickness + 1)

        # Label
        label = f"{pred.diagnosis}"
        if pred.confidence > 0:
            label += f" {pred.confidence:.0%}"

        # Background rectangle for text
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1,
        )
        label_y = max(y1 - 8, th + 4)
        cv2.rectangle(
            canvas, (x1, label_y - th - 4), (x1 + tw + 6, label_y + 4),
            color, -1,
        )
        cv2.putText(
            canvas, label, (x1 + 3, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
            cv2.LINE_AA,
        )

        # Optionally draw lesion box
        if show_lesion_boxes and pred.lesion_bbox:
            lx1, ly1, lx2, ly2 = _int_box(pred.lesion_bbox)
            cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), color, 1, cv2.LINE_AA)

    # Add disclaimer watermark
    disclaimer = "RESEARCH PROTOTYPE - NOT FOR CLINICAL USE"
    (dw, dh), _ = cv2.getTextSize(
        disclaimer, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1,
    )
    cv2.putText(
        canvas, disclaimer, (w - dw - 10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1, cv2.LINE_AA,
    )

    return canvas


def render_image_predictions(
    image: np.ndarray,
    img_preds: ImagePredictions,
    **kwargs,
) -> np.ndarray:
    """Convenience wrapper using an ImagePredictions object."""
    return render_overlay(
        image,
        img_preds.predictions,
        all_teeth=img_preds.teeth,
        **kwargs,
    )


def save_overlay(
    image: np.ndarray,
    predictions: List[ToothPrediction],
    output_path: str,
    **kwargs,
) -> str:
    """Render and save overlay to file."""
    import cv2

    canvas = render_overlay(image, predictions, **kwargs)
    cv2.imwrite(output_path, canvas)
    log.info(f"Saved overlay → {output_path}")
    return output_path


def _int_box(bbox: Tuple[float, ...]) -> Tuple[int, int, int, int]:
    """Convert float bbox to int pixel coordinates."""
    return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
