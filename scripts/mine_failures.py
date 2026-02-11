"""Export hard-case failure sets from clinical evaluation reports."""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path


def configure_logging() -> None:
    """Configure command-line logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_report(path: str) -> dict:
    """Load a clinical evaluation report JSON file."""
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def ensure_layout(base_dir: str, bucket: str) -> tuple[str, str]:
    """Create output folder layout and return image/label directories."""
    images_dir = os.path.join(base_dir, bucket, "images")
    labels_dir = os.path.join(base_dir, bucket, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    return images_dir, labels_dir


def unique_name(prefix: str, source_path: str, index: int) -> str:
    """Build stable filename for copied hard-case assets."""
    stem = Path(source_path).stem
    suffix = Path(source_path).suffix
    return f"{prefix}_{index:04d}_{stem}{suffix}"


def copy_case_assets(
    case: dict,
    output_dir: str,
    bucket: str,
    index: int,
) -> dict:
    """Copy image/label pair for one failure case to target bucket."""
    image_path = case.get("image_path", "")
    label_path = case.get("label_path", "")
    images_dir, labels_dir = ensure_layout(output_dir, bucket)

    copied_image = None
    copied_label = None

    if image_path and os.path.isfile(image_path):
        image_name = unique_name(bucket, image_path, index)
        copied_image = os.path.join(images_dir, image_name)
        shutil.copy2(image_path, copied_image)

    if label_path and os.path.isfile(label_path):
        label_name = unique_name(bucket, label_path, index)
        copied_label = os.path.join(labels_dir, label_name)
        shutil.copy2(label_path, copied_label)

    return {
        "source_image": image_path,
        "source_label": label_path,
        "copied_image": copied_image,
        "copied_label": copied_label,
        "issues": case.get("issues", []),
    }


def split_failures(failures: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split report failures into FN and FP lists."""
    fn_cases = []
    fp_cases = []

    for case in failures:
        issues = case.get("issues", [])
        has_fn = any(int(item.get("fn", 0)) > 0 for item in issues)
        has_fp = any(int(item.get("fp", 0)) > 0 for item in issues)
        if has_fn:
            fn_cases.append(case)
        if has_fp:
            fp_cases.append(case)

    return fn_cases, fp_cases


def write_json(path: str, payload: dict) -> None:
    """Write a JSON payload to disk."""
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def run_mining(eval_report: dict, output_dir: str) -> dict:
    """Run hard-case export and return summary payload."""
    failures = eval_report.get("failures", [])
    fn_cases, fp_cases = split_failures(failures)

    fn_manifest = []
    fp_manifest = []

    for idx, case in enumerate(fn_cases, start=1):
        fn_manifest.append(copy_case_assets(case, output_dir, "hard_fn", idx))
    for idx, case in enumerate(fp_cases, start=1):
        fp_manifest.append(copy_case_assets(case, output_dir, "hard_fp", idx))

    summary = {
        "source_failures": len(failures),
        "hard_fn_cases": len(fn_cases),
        "hard_fp_cases": len(fp_cases),
        "output_dir": os.path.abspath(output_dir),
        "hard_fn_manifest": os.path.join(output_dir, "hard_fn_manifest.json"),
        "hard_fp_manifest": os.path.join(output_dir, "hard_fp_manifest.json"),
    }

    write_json(
        summary["hard_fn_manifest"],
        {"cases": fn_manifest, "count": len(fn_manifest)},
    )
    write_json(
        summary["hard_fp_manifest"],
        {"cases": fp_manifest, "count": len(fp_manifest)},
    )
    write_json(os.path.join(output_dir, "summary.json"), summary)
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export hard FN/FP datasets from clinical eval report",
    )
    parser.add_argument(
        "--eval-report",
        required=True,
        help="Path to results/clinical_eval_report.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/hard_cases",
        help="Directory for exported hard-case files",
    )
    args = parser.parse_args()

    configure_logging()
    os.makedirs(args.output_dir, exist_ok=True)

    report = load_report(args.eval_report)
    summary = run_mining(report, args.output_dir)
    logging.info("Exported hard FN cases: %d", summary["hard_fn_cases"])
    logging.info("Exported hard FP cases: %d", summary["hard_fp_cases"])
    logging.info("Hard-case summary: %s", os.path.join(args.output_dir, "summary.json"))


if __name__ == "__main__":
    main()
