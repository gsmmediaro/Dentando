"""Run triage with uncertainty and image-quality gating."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dental_tooth_caries_ai.clinical_ops import configure_logging
from dental_tooth_caries_ai.clinical_ops import image_quality_signals
from dental_tooth_caries_ai.clinical_ops import is_low_quality
from dental_tooth_caries_ai.clinical_ops import load_thresholds_json
from dental_tooth_caries_ai.clinical_ops import map_thresholds_to_class_ids


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_class_names(raw_names: str) -> list[str]:
    """Parse comma-separated class names."""
    return [name.strip() for name in raw_names.split(",") if name.strip()]


def parse_critical_ids(raw_critical: str, class_names: list[str]) -> list[int]:
    """Map critical class names into class IDs."""
    critical_names = parse_class_names(raw_critical)
    ids = []
    for class_name in critical_names:
        if class_name in class_names:
            ids.append(class_names.index(class_name))
    return ids


def collect_input_images(input_path: str) -> list[str]:
    """Collect image paths from a file or directory."""
    path_obj = Path(input_path)
    if path_obj.is_file():
        return [str(path_obj)]

    if path_obj.is_dir():
        return [
            str(path)
            for path in sorted(path_obj.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ]

    raise FileNotFoundError(f"Input path not found: {input_path}")


def classify_findings(
    detections: list[dict],
    per_class_thresholds: dict[int, float],
    uncertain_band: float,
) -> tuple[list[dict], list[dict]]:
    """Split detections into confirmed and uncertain buckets."""
    confirmed = []
    uncertain = []

    for det in detections:
        class_id = int(det["class_id"])
        conf = float(det["confidence"])
        threshold = float(per_class_thresholds.get(class_id, 0.25))
        uncertain_floor = max(0.01, threshold - uncertain_band)

        if conf >= threshold:
            confirmed.append(det)
        elif conf >= uncertain_floor:
            uncertain.append(det)

    return confirmed, uncertain


def draw_overlay(
    image,
    confirmed: list[dict],
    uncertain: list[dict],
    class_names: list[str],
):
    """Draw confirmed and uncertain detections on image."""
    out = image.copy()

    for det in confirmed:
        x1, y1, x2, y2 = [int(value) for value in det["bbox"]]
        class_id = int(det["class_id"])
        score = float(det["confidence"])
        class_name = (
            class_names[class_id] if class_id < len(class_names) else str(class_id)
        )
        label = f"{class_name} {score:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (50, 210, 50), 2)
        cv2.putText(
            out,
            label,
            (x1, max(10, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (50, 210, 50),
            1,
            cv2.LINE_AA,
        )

    for det in uncertain:
        x1, y1, x2, y2 = [int(value) for value in det["bbox"]]
        class_id = int(det["class_id"])
        score = float(det["confidence"])
        class_name = (
            class_names[class_id] if class_id < len(class_names) else str(class_id)
        )
        label = f"uncertain {class_name} {score:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            out,
            label,
            (x1, max(10, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 180, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def run_triage(args: argparse.Namespace) -> dict:
    """Run triage inference and return report payload."""
    class_names = parse_class_names(args.class_names)
    if not class_names:
        raise ValueError("Provide --class-names to map class IDs")

    threshold_payload = load_thresholds_json(args.thresholds_json)
    per_class_thresholds = map_thresholds_to_class_ids(
        base_conf=args.conf,
        class_names=class_names,
        threshold_payload=threshold_payload,
    )
    critical_ids = parse_critical_ids(args.critical_classes, class_names)
    image_paths = collect_input_images(args.input)

    quality_thresholds = {
        "min_blur_var": args.min_blur_var,
        "min_contrast": args.min_contrast,
        "min_brightness": args.min_brightness,
        "max_brightness": args.max_brightness,
    }

    model = YOLO(args.weights)
    cases = []
    summary = {
        "priority_review": 0,
        "needs_review": 0,
        "routine_review": 0,
    }

    overlay_dir = None
    if args.save_overlays:
        overlay_dir = os.path.join(args.save_overlays, "triage_overlays")
        os.makedirs(overlay_dir, exist_ok=True)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            logging.warning("Skipping unreadable image: %s", image_path)
            continue

        quality_signals = image_quality_signals(image)
        quality_failed = is_low_quality(quality_signals, quality_thresholds)

        raw_results = model(
            image, conf=max(0.01, args.conf - args.uncertain_band), verbose=False
        )
        detections = []
        if raw_results and raw_results[0].boxes is not None:
            for box in raw_results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                detections.append(
                    {
                        "class_id": int(box.cls[0]),
                        "confidence": float(box.conf[0]),
                        "bbox": [
                            float(xyxy[0]),
                            float(xyxy[1]),
                            float(xyxy[2]),
                            float(xyxy[3]),
                        ],
                    }
                )

        confirmed, uncertain = classify_findings(
            detections,
            per_class_thresholds,
            args.uncertain_band,
        )

        reasons = []
        if quality_failed:
            reasons.append("low_image_quality")
        if uncertain:
            reasons.append("uncertain_detections")

        has_critical_confirmed = any(
            int(item["class_id"]) in critical_ids for item in confirmed
        )

        if quality_failed or uncertain:
            decision = "needs_review"
        elif has_critical_confirmed:
            decision = "priority_review"
        else:
            if args.negative_policy == "review":
                decision = "needs_review"
                reasons.append("negative_case_manual_review")
            else:
                decision = "routine_review"

        summary[decision] += 1

        case = {
            "image_path": image_path,
            "decision": decision,
            "reasons": reasons,
            "quality": {
                "signals": quality_signals,
                "failed": quality_failed,
            },
            "confirmed": confirmed,
            "uncertain": uncertain,
        }
        cases.append(case)

        if overlay_dir:
            overlay = draw_overlay(image, confirmed, uncertain, class_names)
            output_path = os.path.join(
                overlay_dir,
                f"{Path(image_path).stem}_triage.jpg",
            )
            cv2.imwrite(output_path, overlay)

    total_cases = len(cases)
    uncertain_cases = sum(1 for case in cases if case["decision"] == "needs_review")
    uncertain_rate = uncertain_cases / max(1, total_cases)

    return {
        "meta": {
            "weights": os.path.abspath(args.weights),
            "input": os.path.abspath(args.input),
            "num_cases": total_cases,
            "class_names": class_names,
            "critical_classes": [class_names[idx] for idx in critical_ids],
            "default_conf": args.conf,
            "uncertain_band": args.uncertain_band,
            "negative_policy": args.negative_policy,
            "thresholds": {
                str(k): round(float(v), 4) for k, v in per_class_thresholds.items()
            },
            "quality_thresholds": quality_thresholds,
        },
        "summary": {
            "priority_review": summary["priority_review"],
            "needs_review": summary["needs_review"],
            "routine_review": summary["routine_review"],
            "needs_review_rate": round(uncertain_rate, 4),
        },
        "cases": cases,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Triage dental predictions with uncertainty checks",
    )
    parser.add_argument("--weights", required=True, help="Path to YOLO weights")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument(
        "--class-names", required=True, help="Comma-separated class names"
    )
    parser.add_argument(
        "--critical-classes", default="", help="Comma-separated critical class names"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Default per-class threshold"
    )
    parser.add_argument(
        "--thresholds-json", default=None, help="Optional thresholds JSON"
    )
    parser.add_argument(
        "--uncertain-band",
        type=float,
        default=0.1,
        help="Confidence band below threshold",
    )
    parser.add_argument(
        "--negative-policy",
        default="review",
        choices=["review", "routine"],
        help="How to route negative cases",
    )
    parser.add_argument(
        "--min-blur-var", type=float, default=60.0, help="Minimum Laplacian variance"
    )
    parser.add_argument(
        "--min-contrast", type=float, default=20.0, help="Minimum grayscale std"
    )
    parser.add_argument(
        "--min-brightness", type=float, default=35.0, help="Minimum mean grayscale"
    )
    parser.add_argument(
        "--max-brightness", type=float, default=220.0, help="Maximum mean grayscale"
    )
    parser.add_argument(
        "--save-overlays", default=None, help="Optional output directory for overlays"
    )
    parser.add_argument(
        "--output",
        default="results/triage_report.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    configure_logging()
    payload = run_triage(args)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)

    logging.info("Saved triage report: %s", args.output)
    logging.info("Needs-review rate: %.4f", payload["summary"]["needs_review_rate"])


if __name__ == "__main__":
    main()
