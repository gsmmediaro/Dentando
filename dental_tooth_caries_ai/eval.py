#!/usr/bin/env python3
"""
Evaluate trained YOLO dental caries models.

Outputs:
    1. Standard mAP metrics via Ultralytics model.val()
    2. Tooth-level metrics (precision/recall/F1) after lesion→tooth assignment
    3. JSON report saved to results/
    4. Rendered overlay examples saved to results/examples/

Usage:
    python -m dental_tooth_caries_ai.eval --modality pano \\
        [--weights runs/dental/pano_caries_only/weights/best.pt] \\
        [--output-dir results]
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_YAML_PATHS = {
    "pano": "data/dentex/yolo/data.yaml",
    "bitewing": "data/bitewing/yolo/data.yaml",
}

DEFAULT_WEIGHTS = {
    "pano": [
        "runs/detect/runs/dental/pano_caries_only/weights/best.pt",
        "runs/dental/pano_caries_only/weights/best.pt",
    ],
    "bitewing": [
        "runs/detect/runs/dental/bitewing_caries_only/weights/best.pt",
        "runs/dental/bitewing_caries_only/weights/best.pt",
    ],
}

# Class names per mode
CLASS_NAMES = {
    "pano_caries_only": ["Caries", "Deep Caries"],
    "pano_caries_plus": ["Caries", "Deep Caries", "Periapical Lesion", "Impacted"],
    "bitewing_caries_only": ["caries"],
}


def run_yolo_val(
    weights_path: str, data_yaml: str, imgsz: int = 640, device: str = "0"
):
    """Run Ultralytics YOLO validation and return metrics."""
    from ultralytics import YOLO

    model = YOLO(weights_path)
    results = model.val(data=data_yaml, imgsz=imgsz, device=device)
    return model, results


def compute_tooth_level_metrics(
    predictions_per_image: list,
    ground_truth_per_image: list,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Compute tooth-level precision, recall, and F1 score.

    For DENTEX (panoramic), each prediction IS a tooth-level prediction.
    We match predicted tooth boxes to ground truth tooth boxes by IoU.

    Args:
        predictions_per_image: List of lists of dicts with "bbox", "class_id", "confidence".
        ground_truth_per_image: List of lists of dicts with "bbox", "class_id".
        iou_threshold: IoU threshold for matching.

    Returns:
        dict with per-class and overall precision, recall, F1.
    """
    from dental_tooth_caries_ai.tooth_level.assign_lesions_to_teeth import compute_iou

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for preds, gts in zip(predictions_per_image, ground_truth_per_image):
        gt_matched = [False] * len(gts)

        # Sort predictions by confidence (descending)
        sorted_preds = sorted(preds, key=lambda x: x.get("confidence", 0), reverse=True)

        for pred in sorted_preds:
            pred_box = pred["bbox"]
            pred_cls = pred["class_id"]
            best_iou = 0
            best_gt_idx = -1

            for j, gt in enumerate(gts):
                if gt_matched[j]:
                    continue
                if gt["class_id"] != pred_cls:
                    continue
                iou = compute_iou(pred_box, gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[pred_cls] += 1
                gt_matched[best_gt_idx] = True
            else:
                fp[pred_cls] += 1

        for j, matched in enumerate(gt_matched):
            if not matched:
                fn[gts[j]["class_id"]] += 1

    # Compute per-class metrics
    all_classes = set(tp.keys()) | set(fp.keys()) | set(fn.keys())
    metrics = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for cls in sorted(all_classes):
        t, f, n = tp[cls], fp[cls], fn[cls]
        precision = t / (t + f) if (t + f) > 0 else 0.0
        recall = t / (t + n) if (t + n) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        metrics[f"class_{cls}"] = {
            "tp": t,
            "fp": f,
            "fn": n,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        total_tp += t
        total_fp += f
        total_fn += n

    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    metrics["overall"] = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
    }

    return metrics


def save_example_overlays(
    model,
    data_yaml: str,
    output_dir: str,
    class_names: list,
    num_examples: int = 10,
    modality: str = "pano",
):
    """Run inference on val images and save overlay examples."""
    import cv2
    import glob
    import yaml

    from dental_tooth_caries_ai.tooth_level.assign_lesions_to_teeth import (
        make_direct_tooth_predictions,
    )
    from dental_tooth_caries_ai.tooth_level.render_overlays import render_overlay

    examples_dir = os.path.join(output_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    # Find val images
    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)

    val_dir = os.path.join(data_cfg["path"], data_cfg["val"])
    val_images = (
        glob.glob(os.path.join(val_dir, "*.jpg"))
        + glob.glob(os.path.join(val_dir, "*.png"))
        + glob.glob(os.path.join(val_dir, "*.jpeg"))
    )

    if not val_images:
        log.warning(f"No val images found in {val_dir}")
        return

    for img_path in val_images[:num_examples]:
        img = cv2.imread(img_path)
        if img is None:
            continue

        results = model(img, conf=0.25)
        if not results or len(results[0].boxes) == 0:
            continue

        # Convert to tooth predictions
        detections = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            detections.append(
                {
                    "bbox": (
                        float(xyxy[0]),
                        float(xyxy[1]),
                        float(xyxy[2]),
                        float(xyxy[3]),
                    ),
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                }
            )

        tooth_preds = make_direct_tooth_predictions(detections, class_names)
        annotated = render_overlay(img, tooth_preds)

        out_name = os.path.splitext(os.path.basename(img_path))[0] + "_overlay.jpg"
        out_path = os.path.join(examples_dir, out_name)
        cv2.imwrite(out_path, annotated)
        log.info(f"  Overlay → {out_path}")


def evaluate(
    modality: str,
    weights: str | None = None,
    data_yaml: str | None = None,
    output_dir: str = "results",
    classes: str = "caries_only",
    imgsz: int = 640,
    device: str = "0",
):
    """Full evaluation pipeline."""

    def resolve_default_weights(modality_name: str) -> str | None:
        """Return first existing default weights path for a modality."""
        candidates = DEFAULT_WEIGHTS.get(modality_name, [])
        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0] if candidates else None

    # Resolve paths
    if weights is None:
        weights = resolve_default_weights(modality)
    if data_yaml is None:
        data_yaml = DATA_YAML_PATHS.get(modality)

    if not weights or not os.path.isfile(weights):
        log.error(f"Model weights not found: {weights}")
        log.info("Train a model first: make train MODALITY=...")
        sys.exit(1)
    if not data_yaml or not os.path.isfile(data_yaml):
        log.error(f"data.yaml not found: {data_yaml}")
        sys.exit(1)

    assert weights is not None
    assert data_yaml is not None

    os.makedirs(output_dir, exist_ok=True)

    # 1. Standard YOLO mAP
    log.info("=" * 60)
    log.info("Running YOLO validation for mAP metrics ...")
    model, val_results = run_yolo_val(weights, data_yaml, imgsz, device)

    # 2. Save report
    class_key = f"{modality}_{classes}"
    class_names = CLASS_NAMES.get(class_key, ["unknown"])

    report = {
        "modality": modality,
        "classes": classes,
        "weights": weights,
        "data_yaml": data_yaml,
        "yolo_metrics": {
            "mAP50": float(val_results.box.map50)
            if hasattr(val_results.box, "map50")
            else None,
            "mAP50-95": float(val_results.box.map)
            if hasattr(val_results.box, "map")
            else None,
        },
        "class_names": class_names,
    }

    report_path = os.path.join(output_dir, f"{modality}_{classes}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved → {report_path}")

    # 3. Save example overlays
    log.info("Generating overlay examples ...")
    save_example_overlays(model, data_yaml, output_dir, class_names, modality=modality)

    # Print summary
    print("\n" + "=" * 60)
    print(f"  Evaluation Report — {modality} ({classes})")
    print("=" * 60)
    print(f"  mAP@50:    {report['yolo_metrics']['mAP50']}")
    print(f"  mAP@50-95: {report['yolo_metrics']['mAP50-95']}")
    print(f"  Report:    {report_path}")
    print(f"  Examples:  {os.path.join(output_dir, 'examples')}/")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate dental caries model")
    parser.add_argument("--modality", required=True, choices=["pano", "bitewing"])
    parser.add_argument("--weights", default=None)
    parser.add_argument("--data-yaml", default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--classes", default="caries_only", choices=["caries_only", "caries_plus"]
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    evaluate(
        modality=args.modality,
        weights=args.weights,
        data_yaml=args.data_yaml,
        output_dir=args.output_dir,
        classes=args.classes,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
