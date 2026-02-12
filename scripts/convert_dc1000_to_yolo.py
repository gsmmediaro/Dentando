"""Convert DC1000 segmentation masks (PNG) to YOLO bounding box labels (txt).

Reads binary masks (0=background, 255=caries), extracts contours,
computes bounding boxes, and writes YOLO-format label files.
Also creates train/val split and data.yaml.
"""

import os
import shutil
import random
import yaml
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DC1000 = ROOT / "DC1000_dataset" / "DC1000_dataset"
OUTPUT = ROOT / "data" / "dc1000" / "yolo"

TRAIN_SRC_IMAGES = DC1000 / "train" / "images"
TRAIN_SRC_LABELS = DC1000 / "train" / "labels"
TEST_SRC_IMAGES = DC1000 / "org_test_dataset" / "images"
TEST_SRC_LABELS = DC1000 / "org_test_dataset" / "labels"

VAL_RATIO = 0.15  # 15% of train goes to val
SEED = 42
MIN_CONTOUR_AREA = 25  # ignore tiny noise blobs


def mask_to_yolo_boxes(mask_path, img_w, img_h):
    """Extract bounding boxes from a binary mask and return YOLO-format lines."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    # Resize mask to match image dimensions if needed
    mh, mw = mask.shape[:2]
    if (mw, mh) != (img_w, img_h):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    # Threshold to binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Convert to YOLO format: class x_center y_center width height (normalized)
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
    return lines


def process_split(src_images, src_labels, dst_images, dst_labels):
    """Convert one split: copy images and generate YOLO label txts."""
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted(src_images.glob("*.png"))
    stats = {"total": 0, "with_boxes": 0, "total_boxes": 0, "skipped": 0}

    for img_path in image_files:
        stem = img_path.stem
        mask_path = src_labels / f"{stem}.png"
        if not mask_path.exists():
            stats["skipped"] += 1
            continue

        # Read image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            stats["skipped"] += 1
            continue
        img_h, img_w = img.shape[:2]

        boxes = mask_to_yolo_boxes(mask_path, img_w, img_h)

        # Copy image
        shutil.copy2(img_path, dst_images / img_path.name)

        # Write label
        label_file = dst_labels / f"{stem}.txt"
        with open(label_file, "w") as f:
            f.write("\n".join(boxes))

        stats["total"] += 1
        if boxes:
            stats["with_boxes"] += 1
            stats["total_boxes"] += len(boxes)

    return stats


def main():
    random.seed(SEED)

    # Clean output
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    # --- Process training data and split into train/val ---
    print("Processing training data...")
    all_train_images = sorted(TRAIN_SRC_IMAGES.glob("*.png"))
    random.shuffle(all_train_images)
    val_count = int(len(all_train_images) * VAL_RATIO)
    val_stems = {p.stem for p in all_train_images[:val_count]}

    # Process train
    train_stats = {"total": 0, "with_boxes": 0, "total_boxes": 0, "skipped": 0}
    val_stats = {"total": 0, "with_boxes": 0, "total_boxes": 0, "skipped": 0}

    for img_path in all_train_images:
        stem = img_path.stem
        mask_path = TRAIN_SRC_LABELS / f"{stem}.png"
        if not mask_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        boxes = mask_to_yolo_boxes(mask_path, img_w, img_h)

        if stem in val_stems:
            dst_img = OUTPUT / "images" / "val"
            dst_lbl = OUTPUT / "labels" / "val"
            s = val_stats
        else:
            dst_img = OUTPUT / "images" / "train"
            dst_lbl = OUTPUT / "labels" / "train"
            s = train_stats

        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img_path, dst_img / img_path.name)
        with open(dst_lbl / f"{stem}.txt", "w") as f:
            f.write("\n".join(boxes))

        s["total"] += 1
        if boxes:
            s["with_boxes"] += 1
            s["total_boxes"] += len(boxes)

    # --- Process test data ---
    print("Processing test data...")
    test_stats = process_split(
        TEST_SRC_IMAGES, TEST_SRC_LABELS,
        OUTPUT / "images" / "test", OUTPUT / "labels" / "test"
    )

    # --- Create data.yaml ---
    data_yaml = {
        "path": str(OUTPUT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["Caries"],
    }
    yaml_path = OUTPUT / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"\nConversion complete!")
    print(f"  Train: {train_stats['total']} images, {train_stats['total_boxes']} boxes ({train_stats['with_boxes']} images with caries)")
    print(f"  Val:   {val_stats['total']} images, {val_stats['total_boxes']} boxes ({val_stats['with_boxes']} images with caries)")
    print(f"  Test:  {test_stats['total']} images, {test_stats['total_boxes']} boxes ({test_stats['with_boxes']} images with caries)")
    print(f"\ndata.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
