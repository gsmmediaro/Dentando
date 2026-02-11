"""Calibrate per-class decision thresholds for dental YOLO models."""

import argparse
import json
import logging
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dental_tooth_caries_ai.clinical_ops import configure_logging
from dental_tooth_caries_ai.clinical_ops import counts_to_metrics
from dental_tooth_caries_ai.clinical_ops import evaluate_image_counts
from dental_tooth_caries_ai.clinical_ops import get_split_images
from dental_tooth_caries_ai.clinical_ops import image_to_label_path
from dental_tooth_caries_ai.clinical_ops import init_class_counts
from dental_tooth_caries_ai.clinical_ops import load_data_config
from dental_tooth_caries_ai.clinical_ops import merge_counts
from dental_tooth_caries_ai.clinical_ops import read_yolo_labels
from dental_tooth_caries_ai.clinical_ops import run_model_on_image


def evaluate_class_threshold(
    preds_per_image: list[list[dict]],
    gts_per_image: list[list[dict]],
    class_id: int,
    threshold: float,
    iou: float,
) -> dict:
    """Compute precision/recall/F1 for one class at one threshold."""
    total_counts = init_class_counts([class_id])

    for preds, gts in zip(preds_per_image, gts_per_image):
        counts = evaluate_image_counts(
            predictions=preds,
            ground_truth=gts,
            class_ids=[class_id],
            iou_threshold=iou,
            per_class_thresholds={class_id: threshold},
        )
        merge_counts(total_counts, counts)

    metrics = counts_to_metrics(total_counts[class_id])
    metrics["threshold"] = threshold
    return metrics


def choose_threshold(
    points: list[dict],
    target_recall: float,
) -> tuple[dict, str]:
    """Choose threshold by recall-first policy, fallback to best F1."""
    valid_points = sorted(points, key=lambda item: item["threshold"])
    meet_target = [pt for pt in valid_points if pt["recall"] >= target_recall]

    if meet_target:
        best = sorted(
            meet_target,
            key=lambda item: (
                item["precision"],
                item["recall"],
                item["threshold"],
            ),
            reverse=True,
        )[0]
        return best, "target_recall"

    best = sorted(
        valid_points,
        key=lambda item: (
            item["f1"],
            item["recall"],
            item["precision"],
        ),
        reverse=True,
    )[0]
    return best, "best_f1_fallback"


def calibrate(args: argparse.Namespace) -> dict:
    """Run calibration across threshold grid for each class."""
    data_cfg = load_data_config(args.data_yaml)
    image_dir, image_paths = get_split_images(data_cfg, args.split)
    class_names = data_cfg["class_names"]

    model = YOLO(args.weights)
    preds_per_image = []
    gts_per_image = []

    logging.info("Collecting predictions for %d images", len(image_paths))
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            logging.warning("Skipping unreadable image: %s", image_path)
            continue

        height, width = image.shape[:2]
        label_path = image_to_label_path(image_path, image_dir)
        gts = read_yolo_labels(label_path, width, height)
        preds = run_model_on_image(model, image, conf=args.min_prediction_conf)
        preds_per_image.append(preds)
        gts_per_image.append(gts)

    grid = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)

    thresholds_by_id = {}
    thresholds_by_name = {}
    curves = {}

    for class_id, class_name in enumerate(class_names):
        points = []
        for threshold in grid:
            metrics = evaluate_class_threshold(
                preds_per_image=preds_per_image,
                gts_per_image=gts_per_image,
                class_id=class_id,
                threshold=float(threshold),
                iou=args.iou,
            )
            points.append(
                {
                    "threshold": round(float(metrics["threshold"]), 4),
                    "precision": round(float(metrics["precision"]), 4),
                    "recall": round(float(metrics["recall"]), 4),
                    "f1": round(float(metrics["f1"]), 4),
                    "tp": int(metrics["tp"]),
                    "fp": int(metrics["fp"]),
                    "fn": int(metrics["fn"]),
                }
            )

        selected, policy = choose_threshold(points, args.target_recall)
        thresholds_by_id[str(class_id)] = round(float(selected["threshold"]), 4)
        thresholds_by_name[class_name] = round(float(selected["threshold"]), 4)
        curves[class_name] = {
            "selection_policy": policy,
            "selected": selected,
            "points": points,
        }

    return {
        "meta": {
            "weights": os.path.abspath(args.weights),
            "data_yaml": os.path.abspath(args.data_yaml),
            "split": args.split,
            "iou": args.iou,
            "target_recall": args.target_recall,
            "threshold_grid": {
                "min": args.threshold_min,
                "max": args.threshold_max,
                "steps": args.threshold_steps,
            },
            "num_images": len(preds_per_image),
        },
        "thresholds_by_class_id": thresholds_by_id,
        "thresholds_by_class_name": thresholds_by_name,
        "curves": curves,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Calibrate class-wise confidence thresholds",
    )
    parser.add_argument("--weights", required=True, help="Path to YOLO weights")
    parser.add_argument("--data-yaml", required=True, help="Path to YOLO data yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--iou", type=float, default=0.5, help="IoU matching threshold")
    parser.add_argument(
        "--target-recall", type=float, default=0.9, help="Minimum desired recall"
    )
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=19)
    parser.add_argument("--min-prediction-conf", type=float, default=0.01)
    parser.add_argument(
        "--output",
        default="results/calibrated_thresholds.json",
        help="Output calibration JSON path",
    )
    args = parser.parse_args()

    configure_logging()
    payload = calibrate(args)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)

    logging.info("Saved calibration file: %s", args.output)
    for class_name, threshold in payload["thresholds_by_class_name"].items():
        logging.info("Threshold %-20s %.3f", class_name, threshold)


if __name__ == "__main__":
    main()
