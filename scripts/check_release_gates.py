"""Check clinical release gates against evaluation outputs."""

import argparse
import json
import logging
import os
import sys

import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dental_tooth_caries_ai.clinical_ops import configure_logging


def load_json(path: str) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_yaml(path: str) -> dict:
    """Load a YAML file."""
    with open(path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def check_min(metric_name: str, actual: float, minimum: float) -> dict:
    """Evaluate one minimum threshold gate."""
    passed = actual >= minimum
    return {
        "metric": metric_name,
        "actual": round(float(actual), 4),
        "required": round(float(minimum), 4),
        "passed": passed,
    }


def check_max(metric_name: str, actual: float, maximum: float) -> dict:
    """Evaluate one maximum threshold gate."""
    passed = actual <= maximum
    return {
        "metric": metric_name,
        "actual": round(float(actual), 4),
        "required": round(float(maximum), 4),
        "passed": passed,
    }


def run_checks(eval_report: dict, gates: dict, triage_report: dict | None) -> dict:
    """Run all configured gates and return a consolidated result."""
    checks = []
    metrics = eval_report["metrics"]

    overall_cfg = gates.get("overall", {})
    if "min_macro_recall" in overall_cfg:
        checks.append(
            check_min(
                "macro_recall",
                metrics["macro"]["recall"],
                overall_cfg["min_macro_recall"],
            )
        )
    if "min_macro_precision" in overall_cfg:
        checks.append(
            check_min(
                "macro_precision",
                metrics["macro"]["precision"],
                overall_cfg["min_macro_precision"],
            )
        )
    if "min_overall_recall" in overall_cfg:
        checks.append(
            check_min(
                "overall_recall",
                metrics["overall"]["recall"],
                overall_cfg["min_overall_recall"],
            )
        )
    if "max_failure_rate" in overall_cfg:
        checks.append(
            check_max(
                "failure_rate",
                eval_report["failure_mining"]["error_rate"],
                overall_cfg["max_failure_rate"],
            )
        )

    for class_gate in gates.get("critical_classes", []):
        class_name = class_gate.get("name")
        class_metric = None
        for class_payload in metrics["class_metrics"].values():
            if class_payload["name"] == class_name:
                class_metric = class_payload
                break

        if class_metric is None:
            checks.append(
                {
                    "metric": f"{class_name}_presence",
                    "actual": "missing",
                    "required": "present",
                    "passed": False,
                }
            )
            continue

        if "min_recall" in class_gate:
            checks.append(
                check_min(
                    f"{class_name}_recall",
                    class_metric["recall"],
                    class_gate["min_recall"],
                )
            )
        if "min_precision" in class_gate:
            checks.append(
                check_min(
                    f"{class_name}_precision",
                    class_metric["precision"],
                    class_gate["min_precision"],
                )
            )

    if triage_report is not None:
        triage_cfg = gates.get("triage", {})
        if "max_needs_review_rate" in triage_cfg:
            checks.append(
                check_max(
                    "needs_review_rate",
                    triage_report["summary"]["needs_review_rate"],
                    triage_cfg["max_needs_review_rate"],
                )
            )

    passed = all(item["passed"] for item in checks)
    return {
        "passed": passed,
        "checks": checks,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Check release gate criteria")
    parser.add_argument(
        "--eval-report", required=True, help="Path to evaluate_clinical report"
    )
    parser.add_argument("--gates", required=True, help="Path to release gates yml")
    parser.add_argument(
        "--triage-report", default=None, help="Optional triage report json"
    )
    parser.add_argument(
        "--output",
        default="results/release_gate_report.json",
        help="Output gate report JSON path",
    )
    args = parser.parse_args()

    configure_logging()
    eval_report = load_json(args.eval_report)
    gates_cfg = load_yaml(args.gates)
    triage_report = load_json(args.triage_report) if args.triage_report else None

    result = run_checks(eval_report, gates_cfg, triage_report)
    payload = {
        "meta": {
            "eval_report": os.path.abspath(args.eval_report),
            "gates": os.path.abspath(args.gates),
            "triage_report": os.path.abspath(args.triage_report)
            if args.triage_report
            else None,
        },
        "result": result,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)

    logging.info("Saved gate check report: %s", args.output)
    if result["passed"]:
        logging.info("Release status: PASS")
    else:
        logging.error("Release status: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
