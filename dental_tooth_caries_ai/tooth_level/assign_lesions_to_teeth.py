#!/usr/bin/env python3
"""
Assign lesion detections to tooth instances via IoU overlap.

This module is used for the **bitewing** pathway where the model detects
caries lesion boxes, and we want to map them to tooth instances.

For DENTEX (panoramic), the detections are already tooth-level — use
them directly without this mapping step.

Strategy:
    1. Compute IoU between each lesion box and each tooth box.
    2. Assign lesion to tooth with maximum IoU (if IoU >= threshold).
    3. Fallback: if max IoU = 0, assign to nearest centroid.
    4. A tooth can receive multiple lesions (worst diagnosis wins).
"""

import logging
from typing import List, Optional, Tuple

from .tooth_instance import ToothInstance, ToothPrediction

log = logging.getLogger(__name__)

# Minimum IoU to consider a valid overlap
IOU_THRESHOLD = 0.05


def compute_iou(box_a: Tuple[float, ...], box_b: Tuple[float, ...]) -> float:
    """
    Compute IoU between two boxes in (x1, y1, x2, y2) format.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def centroid_distance(box_a: Tuple[float, ...], box_b: Tuple[float, ...]) -> float:
    """Euclidean distance between centroids of two boxes."""
    cx_a = (box_a[0] + box_a[2]) / 2
    cy_a = (box_a[1] + box_a[3]) / 2
    cx_b = (box_b[0] + box_b[2]) / 2
    cy_b = (box_b[1] + box_b[3]) / 2
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


def assign_lesions_to_teeth(
    teeth: List[ToothInstance],
    lesion_boxes: List[Tuple[float, float, float, float]],
    lesion_classes: List[str],
    lesion_confidences: Optional[List[float]] = None,
    iou_threshold: float = IOU_THRESHOLD,
) -> List[ToothPrediction]:
    """
    Map lesion detections to tooth instances.

    Args:
        teeth: List of tooth instances (with bounding boxes).
        lesion_boxes: List of lesion bounding boxes [(x1, y1, x2, y2), ...].
        lesion_classes: Corresponding class labels ["caries", ...].
        lesion_confidences: Optional confidence scores for each lesion.
        iou_threshold: Minimum IoU to consider a match.

    Returns:
        List of ToothPrediction, one per matched lesion→tooth pair.
    """
    if lesion_confidences is None:
        lesion_confidences = [1.0] * len(lesion_boxes)

    if not teeth:
        log.warning("No tooth instances provided — cannot assign lesions.")
        return []

    predictions = []
    unmatched = 0

    for i, (lbox, lclass, lconf) in enumerate(
        zip(lesion_boxes, lesion_classes, lesion_confidences)
    ):
        # Compute IoU with all teeth
        ious = [(t, compute_iou(lbox, t.bbox)) for t in teeth]
        best_tooth, best_iou = max(ious, key=lambda x: x[1])

        if best_iou >= iou_threshold:
            predictions.append(
                ToothPrediction(
                    tooth=best_tooth,
                    diagnosis=lclass,
                    confidence=lconf,
                    source="mapped",
                    lesion_bbox=lbox,
                )
            )
        else:
            # Fallback: nearest centroid
            dists = [(t, centroid_distance(lbox, t.bbox)) for t in teeth]
            nearest_tooth, nearest_dist = min(dists, key=lambda x: x[1])
            predictions.append(
                ToothPrediction(
                    tooth=nearest_tooth,
                    diagnosis=lclass,
                    confidence=lconf * 0.5,  # Reduced confidence for fallback
                    source="mapped_centroid_fallback",
                    lesion_bbox=lbox,
                )
            )
            unmatched += 1

    if unmatched:
        log.info(
            f"Assigned {len(predictions)} lesions: "
            f"{len(predictions) - unmatched} by IoU, {unmatched} by centroid fallback."
        )

    return predictions


def make_direct_tooth_predictions(
    detections: List[dict],
    class_names: List[str],
) -> List[ToothPrediction]:
    """
    Convert YOLO detections directly to tooth predictions (DENTEX pathway).

    Each detection IS a tooth-level prediction — no mapping needed.

    Args:
        detections: List of dicts with keys: "bbox" (x1,y1,x2,y2),
                    "class_id", "confidence".
        class_names: List of class name strings indexed by class_id.

    Returns:
        List of ToothPrediction.
    """
    predictions = []
    for i, det in enumerate(detections):
        bbox = tuple(det["bbox"])
        cls_id = det["class_id"]
        conf = det.get("confidence", 1.0)

        tooth = ToothInstance(
            tooth_id=i,
            bbox=bbox,
            confidence=conf,
        )
        diagnosis = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

        predictions.append(
            ToothPrediction(
                tooth=tooth,
                diagnosis=diagnosis,
                confidence=conf,
                source="direct",
            )
        )

    return predictions


# ── Self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running assign_lesions_to_teeth self-test ...")

    teeth = [
        ToothInstance(0, (10, 10, 50, 50)),
        ToothInstance(1, (60, 10, 100, 50)),
        ToothInstance(2, (110, 10, 150, 50)),
    ]

    lesions = [
        (15, 15, 45, 45),   # overlaps tooth 0
        (65, 15, 95, 45),   # overlaps tooth 1
        (200, 200, 220, 220),  # no overlap, fallback to nearest
    ]
    classes = ["caries", "caries", "deep_caries"]

    preds = assign_lesions_to_teeth(teeth, lesions, classes)

    assert len(preds) == 3
    assert preds[0].tooth.tooth_id == 0
    assert preds[1].tooth.tooth_id == 1
    assert preds[2].source == "mapped_centroid_fallback"

    print(f"✓ All tests passed — {len(preds)} predictions generated.")
    for p in preds:
        print(f"  Tooth {p.tooth.tooth_id}: {p.diagnosis} "
              f"(conf={p.confidence:.2f}, source={p.source})")
