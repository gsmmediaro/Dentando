"""Clinical-style evaluation for dental YOLO models."""

import argparse
import json
import logging
import os
import sys

import cv2
from ultralytics import YOLO

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dental_tooth_caries_ai.clinical_ops import bootstrap_metrics
from dental_tooth_caries_ai.clinical_ops import configure_logging
from dental_tooth_caries_ai.clinical_ops import counts_to_metrics
from dental_tooth_caries_ai.clinical_ops import evaluate_image_counts
from dental_tooth_caries_ai.clinical_ops import get_split_images
from dental_tooth_caries_ai.clinical_ops import image_to_label_path
from dental_tooth_caries_ai.clinical_ops import init_class_counts
from dental_tooth_caries_ai.clinical_ops import load_data_config
from dental_tooth_caries_ai.clinical_ops import load_thresholds_json
from dental_tooth_caries_ai.clinical_ops import map_thresholds_to_class_ids
from dental_tooth_caries_ai.clinical_ops import merge_counts
from dental_tooth_caries_ai.clinical_ops import read_yolo_labels
from dental_tooth_caries_ai.clinical_ops import run_model_on_image


def build_report(
    class_names: list[str],
    total_counts: dict[int, dict[str, int]],
    ci_by_class: dict[int, dict[str, tuple[float, float]]],
) -> dict:
    """Build structured report payload from counts and confidence intervals."""
    class_metrics = {}
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for class_id, class_name in enumerate(class_names):
        metrics = counts_to_metrics(total_counts[class_id])
        ci = ci_by_class.get(class_id, {})

        class_metrics[str(class_id)] = {
            "name": class_name,
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "precision": round(float(metrics["precision"]), 4),
            "recall": round(float(metrics["recall"]), 4),
            "f1": round(float(metrics["f1"]), 4),
            "precision_ci95": [
                round(float(ci.get("precision", (0.0, 0.0))[0]), 4),
                round(float(ci.get("precision", (0.0, 0.0))[1]), 4),
            ],
            "recall_ci95": [
                round(float(ci.get("recall", (0.0, 0.0))[0]), 4),
                round(float(ci.get("recall", (0.0, 0.0))[1]), 4),
            ],
            "f1_ci95": [
                round(float(ci.get("f1", (0.0, 0.0))[0]), 4),
                round(float(ci.get("f1", (0.0, 0.0))[1]), 4),
            ],
        }
        all_precisions.append(float(metrics["precision"]))
        all_recalls.append(float(metrics["recall"]))
        all_f1s.append(float(metrics["f1"]))

    overall_counts = {"tp": 0, "fp": 0, "fn": 0}
    for class_id in range(len(class_names)):
        overall_counts["tp"] += int(total_counts[class_id]["tp"])
        overall_counts["fp"] += int(total_counts[class_id]["fp"])
        overall_counts["fn"] += int(total_counts[class_id]["fn"])
    overall_metrics = counts_to_metrics(overall_counts)

    macro_precision = sum(all_precisions) / max(1, len(all_precisions))
    macro_recall = sum(all_recalls) / max(1, len(all_recalls))
    macro_f1 = sum(all_f1s) / max(1, len(all_f1s))

    return {
        "overall": {
            "tp": overall_metrics["tp"],
            "fp": overall_metrics["fp"],
            "fn": overall_metrics["fn"],
            "precision": round(float(overall_metrics["precision"]), 4),
            "recall": round(float(overall_metrics["recall"]), 4),
            "f1": round(float(overall_metrics["f1"]), 4),
        },
        "macro": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4),
        },
        "class_metrics": class_metrics,
    }


def parse_critical_class_ids(
    critical_classes: str,
    class_names: list[str],
) -> list[int]:
    """Parse critical class names into class IDs."""
    if not critical_classes:
        return []

    ids = []
    requested = [name.strip() for name in critical_classes.split(",") if name.strip()]
    for class_name in requested:
        if class_name in class_names:
            ids.append(class_names.index(class_name))
    return ids


def evaluate(args: argparse.Namespace) -> dict:
    """Run clinical-style model evaluation and return full report payload."""
    data_cfg = load_data_config(args.data_yaml)
    image_dir, image_paths = get_split_images(data_cfg, args.split)
    class_names = data_cfg["class_names"]
    class_ids = list(range(len(class_names)))

    threshold_payload = load_thresholds_json(args.thresholds_json)
    thresholds = map_thresholds_to_class_ids(args.conf, class_names, threshold_payload)

    logging.info("Loading model: %s", args.weights)
    model = YOLO(args.weights)

    total_counts = init_class_counts(class_ids)
    per_image_counts = []
    failure_manifest = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            logging.warning("Skipping unreadable image: %s", image_path)
            continue

        height, width = image.shape[:2]
        label_path = image_to_label_path(image_path, image_dir)
        gts = read_yolo_labels(label_path, width, height)
        preds = run_model_on_image(model, image, conf=args.min_prediction_conf)

        counts = evaluate_image_counts(
            predictions=preds,
            ground_truth=gts,
            class_ids=class_ids,
            iou_threshold=args.iou,
            per_class_thresholds=thresholds,
        )
        per_image_counts.append(counts)
        merge_counts(total_counts, counts)

        image_failure = False
        reasons = []
        for class_id in class_ids:
            class_counts = counts[class_id]
            if class_counts["fp"] > 0 or class_counts["fn"] > 0:
                image_failure = True
                reasons.append(
                    {
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "fp": class_counts["fp"],
                        "fn": class_counts["fn"],
                    }
                )
        if image_failure:
            failure_manifest.append(
                {
                    "image_path": image_path,
                    "label_path": label_path,
                    "issues": reasons,
                }
            )

    ci_by_class = bootstrap_metrics(
        image_level_counts=per_image_counts,
        class_ids=class_ids,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )

    metrics = build_report(class_names, total_counts, ci_by_class)
    critical_ids = parse_critical_class_ids(args.critical_classes, class_names)

    critical_summary = {}
    for class_id in critical_ids:
        class_metrics = metrics["class_metrics"][str(class_id)]
        critical_summary[class_metrics["name"]] = {
            "recall": class_metrics["recall"],
            "recall_ci95": class_metrics["recall_ci95"],
            "precision": class_metrics["precision"],
        }

    payload = {
        "meta": {
            "weights": os.path.abspath(args.weights),
            "data_yaml": os.path.abspath(args.data_yaml),
            "split": args.split,
            "iou": args.iou,
            "default_conf": args.conf,
            "min_prediction_conf": args.min_prediction_conf,
            "thresholds": {str(k): round(v, 4) for k, v in thresholds.items()},
            "num_images": len(image_paths),
            "bootstrap_samples": args.bootstrap_samples,
        },
        "metrics": metrics,
        "critical_summary": critical_summary,
        "failure_mining": {
            "num_images_with_errors": len(failure_manifest),
            "error_rate": round(
                len(failure_manifest) / max(1, len(image_paths)),
                4,
            ),
        },
        "failures": failure_manifest,
    }
    return payload


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate dental YOLO model with clinical metrics",
    )
    parser.add_argument("--weights", required=True, help="Path to YOLO weights")
    parser.add_argument("--data-yaml", required=True, help="Path to YOLO data yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--iou", type=float, default=0.5, help="IoU matching threshold")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Default class threshold"
    )
    parser.add_argument(
        "--min-prediction-conf",
        type=float,
        default=0.01,
        help="Low conf floor for raw model predictions",
    )
    parser.add_argument(
        "--thresholds-json", default=None, help="Optional thresholds JSON file"
    )
    parser.add_argument(
        "--bootstrap-samples", type=int, default=300, help="Bootstrap samples for CI"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--critical-classes", default="", help="Comma-separated class names"
    )
    parser.add_argument(
        "--output",
        default="results/clinical_eval_report.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    configure_logging()
    payload = evaluate(args)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)

    logging.info("Saved report: %s", args.output)
    logging.info("Overall recall: %.4f", payload["metrics"]["overall"]["recall"])
    logging.info("Overall precision: %.4f", payload["metrics"]["overall"]["precision"])


if __name__ == "__main__":
    main()
