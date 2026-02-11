"""Utility helpers for clinical-style model evaluation pipelines."""

import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def configure_logging() -> None:
    """Configure default logging for CLI scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_data_config(data_yaml_path: str) -> dict:
    """Load YOLO data YAML and normalize paths and class names."""
    with open(data_yaml_path, "r", encoding="utf-8") as file_obj:
        cfg = yaml.safe_load(file_obj)

    base_path = cfg.get("path")
    if base_path is None:
        base_path = os.path.dirname(os.path.abspath(data_yaml_path))

    if not os.path.isabs(base_path):
        yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))
        base_path = os.path.abspath(os.path.join(yaml_dir, base_path))

    class_names = normalize_class_names(cfg.get("names", []), cfg.get("nc"))

    return {
        "path": base_path,
        "train": cfg.get("train"),
        "val": cfg.get("val"),
        "test": cfg.get("test"),
        "class_names": class_names,
        "nc": len(class_names),
    }


def normalize_class_names(names: object, nc: int | None = None) -> list[str]:
    """Normalize class names from YOLO YAML list/dict forms."""
    if isinstance(names, list):
        normalized = [str(value) for value in names]
    elif isinstance(names, dict):
        keys = sorted(int(key) for key in names.keys())
        normalized = [
            str(names[str(key)] if str(key) in names else names[key]) for key in keys
        ]
    else:
        normalized = []

    if nc is not None and nc > len(normalized):
        for idx in range(len(normalized), nc):
            normalized.append(f"class_{idx}")

    return normalized


def get_split_images(data_cfg: dict, split: str) -> tuple[str, list[str]]:
    """Return absolute split image directory and image file list."""
    split_rel = data_cfg.get(split)
    if not split_rel:
        raise ValueError(f"Split '{split}' not found in data YAML")

    image_dir = split_rel
    if not os.path.isabs(image_dir):
        image_dir = os.path.join(data_cfg["path"], split_rel)
    image_dir = os.path.abspath(image_dir)

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Split directory not found: {image_dir}")

    image_paths = [
        str(path)
        for path in sorted(Path(image_dir).iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]

    return image_dir, image_paths


def image_to_label_path(image_path: str, image_dir: str) -> str:
    """Map a split image path to its YOLO label path."""
    relative_path = os.path.relpath(image_path, image_dir)
    label_dir = image_dir.replace("images", "labels")
    stem, _ = os.path.splitext(relative_path)
    return os.path.join(label_dir, f"{stem}.txt")


def read_yolo_labels(label_path: str, width: int, height: int) -> list[dict]:
    """Read YOLO txt labels and convert boxes to xyxy format."""
    if not os.path.isfile(label_path):
        return []

    labels = []
    with open(label_path, "r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue

            values = line.split()
            if len(values) < 5:
                continue

            class_id = int(float(values[0]))
            center_x = float(values[1]) * width
            center_y = float(values[2]) * height
            box_w = float(values[3]) * width
            box_h = float(values[4]) * height

            x1 = center_x - box_w / 2.0
            y1 = center_y - box_h / 2.0
            x2 = center_x + box_w / 2.0
            y2 = center_y + box_h / 2.0

            labels.append(
                {
                    "class_id": class_id,
                    "bbox": (
                        max(0.0, x1),
                        max(0.0, y1),
                        min(float(width), x2),
                        min(float(height), y2),
                    ),
                }
            )

    return labels


def run_model_on_image(model, image: np.ndarray, conf: float) -> list[dict]:
    """Run YOLO inference and return predictions as dict records."""
    results = model(image, conf=conf, verbose=False)
    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None:
        return []

    predictions = []
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        predictions.append(
            {
                "class_id": int(box.cls[0]),
                "confidence": float(box.conf[0]),
                "bbox": (
                    float(xyxy[0]),
                    float(xyxy[1]),
                    float(xyxy[2]),
                    float(xyxy[3]),
                ),
            }
        )

    return predictions


def compute_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0

    return inter_area / union


def init_class_counts(class_ids: list[int]) -> dict[int, dict[str, int]]:
    """Initialize per-class TP/FP/FN counter dictionary."""
    return {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in class_ids}


def merge_counts(
    destination: dict[int, dict[str, int]],
    source: dict[int, dict[str, int]],
) -> None:
    """Merge source counters into destination in-place."""
    for class_id, values in source.items():
        if class_id not in destination:
            destination[class_id] = {"tp": 0, "fp": 0, "fn": 0}
        destination[class_id]["tp"] += int(values.get("tp", 0))
        destination[class_id]["fp"] += int(values.get("fp", 0))
        destination[class_id]["fn"] += int(values.get("fn", 0))


def evaluate_image_counts(
    predictions: list[dict],
    ground_truth: list[dict],
    class_ids: list[int],
    iou_threshold: float,
    per_class_thresholds: dict[int, float],
) -> dict[int, dict[str, int]]:
    """Compute TP/FP/FN counts for one image with class-specific thresholds."""
    counts = init_class_counts(class_ids)

    for class_id in class_ids:
        threshold = per_class_thresholds.get(class_id, 0.25)
        preds = [
            pred
            for pred in predictions
            if pred["class_id"] == class_id and pred["confidence"] >= threshold
        ]
        gts = [gt for gt in ground_truth if gt["class_id"] == class_id]

        preds = sorted(preds, key=lambda item: item["confidence"], reverse=True)
        matched_gt = [False] * len(gts)

        for pred in preds:
            best_idx = -1
            best_iou = 0.0

            for idx, gt in enumerate(gts):
                if matched_gt[idx]:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx >= 0 and best_iou >= iou_threshold:
                counts[class_id]["tp"] += 1
                matched_gt[best_idx] = True
            else:
                counts[class_id]["fp"] += 1

        for used in matched_gt:
            if not used:
                counts[class_id]["fn"] += 1

    return counts


def safe_divide(numerator: float, denominator: float) -> float:
    """Return 0.0 if denominator is zero."""
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def counts_to_metrics(counts: dict[str, int]) -> dict[str, float | int]:
    """Convert TP/FP/FN counters to precision/recall/F1 metrics."""
    tp = int(counts.get("tp", 0))
    fp = int(counts.get("fp", 0))
    fn = int(counts.get("fn", 0))

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def percentile_interval(
    values: list[float],
    alpha: float = 0.95,
) -> tuple[float, float]:
    """Compute percentile confidence interval for a metric sample list."""
    if not values:
        return 0.0, 0.0
    lower_pct = (1.0 - alpha) / 2.0 * 100.0
    upper_pct = (1.0 + alpha) / 2.0 * 100.0
    low = float(np.percentile(values, lower_pct))
    high = float(np.percentile(values, upper_pct))
    return low, high


def bootstrap_metrics(
    image_level_counts: list[dict[int, dict[str, int]]],
    class_ids: list[int],
    samples: int,
    seed: int,
) -> dict[int, dict[str, tuple[float, float]]]:
    """Bootstrap CIs for per-class precision/recall/F1 using image resampling."""
    if samples <= 0 or not image_level_counts:
        return {}

    rng = np.random.default_rng(seed)
    size = len(image_level_counts)

    bucket = {
        class_id: {"precision": [], "recall": [], "f1": []} for class_id in class_ids
    }

    for _ in range(samples):
        picked = rng.integers(0, size, size=size)
        totals = init_class_counts(class_ids)
        for idx in picked:
            merge_counts(totals, image_level_counts[int(idx)])

        for class_id in class_ids:
            metrics = counts_to_metrics(totals[class_id])
            bucket[class_id]["precision"].append(float(metrics["precision"]))
            bucket[class_id]["recall"].append(float(metrics["recall"]))
            bucket[class_id]["f1"].append(float(metrics["f1"]))

    ci = {}
    for class_id in class_ids:
        ci[class_id] = {
            "precision": percentile_interval(bucket[class_id]["precision"]),
            "recall": percentile_interval(bucket[class_id]["recall"]),
            "f1": percentile_interval(bucket[class_id]["f1"]),
        }

    return ci


def map_thresholds_to_class_ids(
    base_conf: float,
    class_names: list[str],
    threshold_payload: dict | None,
) -> dict[int, float]:
    """Resolve threshold config payload to class_id -> threshold mapping."""
    thresholds = {idx: float(base_conf) for idx in range(len(class_names))}
    if not threshold_payload:
        return thresholds

    by_id = threshold_payload.get("thresholds_by_class_id", {})
    for key, value in by_id.items():
        class_id = int(key)
        thresholds[class_id] = float(value)

    by_name = threshold_payload.get("thresholds_by_class_name", {})
    for class_name, value in by_name.items():
        if class_name in class_names:
            class_id = class_names.index(class_name)
            thresholds[class_id] = float(value)

    return thresholds


def load_thresholds_json(path: str | None) -> dict | None:
    """Load threshold JSON if provided."""
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def image_quality_signals(image: np.ndarray) -> dict[str, float]:
    """Compute simple no-reference quality signals used for triage."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    return {
        "blur_var": blur_var,
        "brightness": brightness,
        "contrast": contrast,
    }


def is_low_quality(signals: dict[str, float], thresholds: dict[str, float]) -> bool:
    """Decide if an image fails quality checks."""
    if signals["blur_var"] < thresholds["min_blur_var"]:
        return True
    if signals["contrast"] < thresholds["min_contrast"]:
        return True
    if signals["brightness"] < thresholds["min_brightness"]:
        return True
    if signals["brightness"] > thresholds["max_brightness"]:
        return True
    return False
