#!/usr/bin/env python3
"""
Train YOLO for dental caries detection.

Wraps Ultralytics training following the conventions of scripts/main.py.

Usage:
    python -m dental_tooth_caries_ai.train --modality pano --task detect \\
        --classes caries_only --epochs 50 --imgsz 640 --batch 16

    python -m dental_tooth_caries_ai.train --modality bitewing --task detect \\
        --classes caries_only --epochs 50 --imgsz 640 --batch 16
"""

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Default data.yaml locations per modality
DATA_YAML_PATHS = {
    "pano": "data/dentex/yolo/data.yaml",
    "bitewing": "data/bitewing/yolo/data.yaml",
    "cbct": "data/cbct/yolo/data.yaml",
}

# Default YOLO model
DEFAULT_MODEL = "yolov8m.pt"


def resolve_data_yaml(modality: str, custom_path: str = None) -> str:
    """Resolve the data.yaml file path."""
    if custom_path and os.path.isfile(custom_path):
        return custom_path
    default = DATA_YAML_PATHS.get(modality)
    if default and os.path.isfile(default):
        return default
    raise FileNotFoundError(
        f"data.yaml not found for modality '{modality}'. "
        f"Expected at: {default}. "
        f"Run 'make prepare DATASET=...' first."
    )


def train(
    modality: str,
    task: str = "detect",
    classes: str = "caries_only",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    weights: str = None,
    resume: bool = False,
    project: str = "runs/dental",
    name: str = None,
    data_yaml: str = None,
):
    """
    Run YOLO training for dental caries detection.

    Args:
        modality: One of "pano", "bitewing", "cbct".
        task: Training task ("detect").
        classes: "caries_only" or "caries_plus".
        epochs: Number of training epochs.
        imgsz: Input image size.
        batch: Batch size.
        device: CUDA device ("0", "cpu", etc.).
        weights: Path to pretrained weights (optional).
        resume: Resume from last checkpoint.
        project: Project directory for runs.
        name: Experiment name (auto-generated if None).
        data_yaml: Custom data.yaml path.
    """
    from ultralytics import YOLO

    yaml_path = resolve_data_yaml(modality, data_yaml)
    log.info(f"Training config: modality={modality}, data={yaml_path}")
    log.info(f"  epochs={epochs}, imgsz={imgsz}, batch={batch}, device={device}")

    if name is None:
        name = f"{modality}_{classes}"

    if resume:
        checkpoint = os.path.join(project, name, "weights", "last.pt")
        if not os.path.exists(checkpoint):
            log.error(f"Checkpoint not found: {checkpoint}")
            sys.exit(1)
        log.info(f"Resuming from {checkpoint}")
        model = YOLO(checkpoint)
        results = model.train(resume=True)
    elif weights:
        log.info(f"Starting from weights: {weights}")
        model = YOLO(weights)
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            exist_ok=True,
        )
    else:
        log.info(f"Starting fresh with {DEFAULT_MODEL}")
        model = YOLO(DEFAULT_MODEL)
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            exist_ok=True,
        )

    log.info(f"Training complete. Results saved to {project}/{name}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO for dental caries detection",
    )
    parser.add_argument("--modality", required=True,
                        choices=["pano", "bitewing", "cbct"],
                        help="Imaging modality")
    parser.add_argument("--task", default="detect",
                        choices=["detect"],
                        help="Training task (default: detect)")
    parser.add_argument("--classes", default="caries_only",
                        choices=["caries_only", "caries_plus"],
                        help="Class set")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--weights", default=None,
                        help="Pretrained weights path")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--project", default="runs/dental")
    parser.add_argument("--name", default=None,
                        help="Experiment name")
    parser.add_argument("--data-yaml", default=None,
                        help="Custom data.yaml path")
    args = parser.parse_args()

    train(
        modality=args.modality,
        task=args.task,
        classes=args.classes,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        weights=args.weights,
        resume=args.resume,
        project=args.project,
        name=args.name,
        data_yaml=args.data_yaml,
    )


if __name__ == "__main__":
    main()
