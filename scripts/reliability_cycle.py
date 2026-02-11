"""Run one reliability improvement cycle for dental YOLO models."""

import argparse
import logging
import os
import shutil
import subprocess
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dental_tooth_caries_ai.clinical_ops import configure_logging
from dental_tooth_caries_ai.clinical_ops import load_data_config


def run_cmd(command: list[str], label: str, allow_failure: bool = False) -> int:
    """Run one subprocess command and optionally tolerate failures."""
    logging.info("[%s] %s", label, " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0 and not allow_failure:
        raise subprocess.CalledProcessError(result.returncode, command)
    if result.returncode != 0 and allow_failure:
        logging.warning("[%s] command exited with code %d", label, result.returncode)
    return int(result.returncode)


def derive_split_dir(data_yaml: str, split: str) -> str:
    """Resolve split image directory from a YOLO data YAML file."""
    data_cfg = load_data_config(data_yaml)
    split_rel = data_cfg.get(split)
    if not split_rel:
        raise ValueError(f"Split not found in data yaml: {split}")
    if os.path.isabs(split_rel):
        return split_rel
    return os.path.join(data_cfg["path"], split_rel)


def derive_class_names_csv(data_yaml: str) -> str:
    """Read class names from data YAML and return comma-separated string."""
    class_names = load_data_config(data_yaml)["class_names"]
    return ",".join(class_names)


def ingest_reviewed_hard_cases(reviewed_dir: str, data_yaml: str) -> int:
    """Copy reviewed hard cases into train split and return copied count."""
    if not reviewed_dir:
        return 0

    reviewed_images = os.path.join(reviewed_dir, "images")
    reviewed_labels = os.path.join(reviewed_dir, "labels")
    if not os.path.isdir(reviewed_images) or not os.path.isdir(reviewed_labels):
        raise FileNotFoundError("Reviewed hard cases must include images/ and labels/")

    data_cfg = load_data_config(data_yaml)
    train_rel = data_cfg.get("train")
    if not train_rel:
        raise ValueError("Train split missing in data yaml")

    train_images = (
        train_rel
        if os.path.isabs(train_rel)
        else os.path.join(data_cfg["path"], train_rel)
    )
    train_labels = train_images.replace("images", "labels")

    os.makedirs(train_images, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)

    copied = 0
    for file_name in sorted(os.listdir(reviewed_images)):
        src_image = os.path.join(reviewed_images, file_name)
        if not os.path.isfile(src_image):
            continue

        stem, ext = os.path.splitext(file_name)
        src_label = os.path.join(reviewed_labels, f"{stem}.txt")
        if not os.path.isfile(src_label):
            continue

        dst_image = os.path.join(train_images, f"hardcase_{stem}{ext}")
        dst_label = os.path.join(train_labels, f"hardcase_{stem}.txt")
        shutil.copy2(src_image, dst_image)
        shutil.copy2(src_label, dst_label)
        copied += 1

    logging.info("Ingested reviewed hard cases into train split: %d", copied)
    return copied


def run_cycle(args: argparse.Namespace) -> None:
    """Execute calibrate -> eval -> mine -> triage -> gate flow."""
    os.makedirs(args.output_dir, exist_ok=True)

    thresholds_path = os.path.join(args.output_dir, "calibrated_thresholds.json")
    eval_path = os.path.join(args.output_dir, "clinical_eval_report.json")
    triage_path = os.path.join(args.output_dir, "triage_report.json")
    gates_path = os.path.join(args.output_dir, "release_gate_report.json")
    hard_cases_dir = os.path.join(args.output_dir, "hard_cases")

    class_names_csv = args.class_names or derive_class_names_csv(args.data_yaml)
    triage_input = args.triage_input or derive_split_dir(args.data_yaml, args.split)

    run_cmd(
        [
            sys.executable,
            os.path.join("scripts", "calibrate_thresholds.py"),
            "--weights",
            args.weights,
            "--data-yaml",
            args.data_yaml,
            "--split",
            args.split,
            "--output",
            thresholds_path,
        ],
        "calibrate",
    )

    run_cmd(
        [
            sys.executable,
            os.path.join("scripts", "evaluate_clinical.py"),
            "--weights",
            args.weights,
            "--data-yaml",
            args.data_yaml,
            "--split",
            args.split,
            "--thresholds-json",
            thresholds_path,
            "--critical-classes",
            args.critical_classes,
            "--output",
            eval_path,
        ],
        "evaluate",
    )

    run_cmd(
        [
            sys.executable,
            os.path.join("scripts", "mine_failures.py"),
            "--eval-report",
            eval_path,
            "--output-dir",
            hard_cases_dir,
        ],
        "mine-failures",
    )

    triage_cmd = [
        sys.executable,
        os.path.join("scripts", "triage_with_uncertainty.py"),
        "--weights",
        args.weights,
        "--input",
        triage_input,
        "--class-names",
        class_names_csv,
        "--critical-classes",
        args.critical_classes,
        "--thresholds-json",
        thresholds_path,
        "--negative-policy",
        args.negative_policy,
        "--output",
        triage_path,
    ]
    if args.save_overlays:
        triage_cmd.extend(["--save-overlays", args.output_dir])
    run_cmd(triage_cmd, "triage")

    gate_code = run_cmd(
        [
            sys.executable,
            os.path.join("scripts", "check_release_gates.py"),
            "--eval-report",
            eval_path,
            "--triage-report",
            triage_path,
            "--gates",
            args.gates,
            "--output",
            gates_path,
        ],
        "release-gates",
        allow_failure=True,
    )
    if gate_code != 0:
        logging.warning("Release gates did not pass for this cycle")

    if args.run_retrain:
        ingest_reviewed_hard_cases(args.reviewed_hard_cases, args.data_yaml)
        retrain_name = args.retrain_name or "reliability_cycle_retrain"

        run_cmd(
            [
                sys.executable,
                "-m",
                "dental_tooth_caries_ai.train",
                "--modality",
                args.modality,
                "--classes",
                args.classes,
                "--epochs",
                str(args.epochs),
                "--imgsz",
                str(args.imgsz),
                "--batch",
                str(args.batch),
                "--device",
                args.device,
                "--weights",
                args.weights,
                "--project",
                args.project,
                "--name",
                retrain_name,
                "--data-yaml",
                args.data_yaml,
            ],
            "retrain",
        )

        retrained_weights = os.path.join(
            args.project,
            retrain_name,
            "weights",
            "best.pt",
        )
        logging.info("Retrained weights: %s", retrained_weights)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run one reliability loop cycle",
    )
    parser.add_argument("--weights", required=True, help="Model weights path")
    parser.add_argument("--data-yaml", required=True, help="YOLO data yaml path")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--critical-classes", default="", help="Critical class names CSV"
    )
    parser.add_argument("--class-names", default="", help="Class names CSV for triage")
    parser.add_argument("--triage-input", default="", help="Image dir/file for triage")
    parser.add_argument(
        "--gates", default="configs/release_gates.yml", help="Release gates YAML path"
    )
    parser.add_argument(
        "--negative-policy", default="review", choices=["review", "routine"]
    )
    parser.add_argument(
        "--output-dir",
        default="results/reliability_cycle",
        help="Output directory for cycle artifacts",
    )
    parser.add_argument(
        "--save-overlays", action="store_true", help="Save triage overlays"
    )

    parser.add_argument(
        "--run-retrain", action="store_true", help="Run retraining stage"
    )
    parser.add_argument(
        "--reviewed-hard-cases", default="", help="Dir with reviewed hard cases"
    )
    parser.add_argument(
        "--modality", default="pano", choices=["pano", "bitewing", "cbct"]
    )
    parser.add_argument(
        "--classes", default="caries_only", choices=["caries_only", "caries_plus"]
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/dental")
    parser.add_argument("--retrain-name", default="")
    args = parser.parse_args()

    configure_logging()
    run_cycle(args)
    logging.info("Reliability cycle complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
