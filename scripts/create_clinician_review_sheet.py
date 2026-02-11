"""Create a clinician review CSV from a hard-case manifest."""

import argparse
import csv
import json
import os
from pathlib import Path


def load_manifest(path: str) -> dict:
    """Load hard-case manifest JSON."""
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def issues_to_text(issues: list[dict]) -> str:
    """Convert issue list into compact text for reviewers."""
    parts = []
    for item in issues:
        class_name = item.get("class_name", "unknown")
        fp_count = int(item.get("fp", 0))
        fn_count = int(item.get("fn", 0))
        parts.append(f"{class_name}: fp={fp_count}, fn={fn_count}")
    return " | ".join(parts)


def create_sheet(manifest_path: str, output_csv: str) -> int:
    """Create clinician review CSV and return number of rows."""
    payload = load_manifest(manifest_path)
    cases = payload.get("cases", [])

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "case_id",
                "image_file",
                "label_file",
                "detected_issue_summary",
                "review_status",
                "clinician_action",
                "corrected_label_file",
                "notes",
            ]
        )

        for index, case in enumerate(cases, start=1):
            image_file = Path(case.get("copied_image", "")).name
            label_file = Path(case.get("copied_label", "")).name
            issue_text = issues_to_text(case.get("issues", []))
            writer.writerow(
                [
                    f"case_{index:03d}",
                    image_file,
                    label_file,
                    issue_text,
                    "pending",
                    "",
                    "",
                    "",
                ]
            )

    return len(cases)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create clinician review CSV from manifest",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to selected_manifest.json or hard_*_manifest.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path",
    )
    args = parser.parse_args()

    count = create_sheet(args.manifest, args.output)
    print(f"Created review sheet with {count} cases: {args.output}")


if __name__ == "__main__":
    main()
