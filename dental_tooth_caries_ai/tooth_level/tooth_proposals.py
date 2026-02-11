#!/usr/bin/env python3
"""
Heuristic tooth proposal for bitewing radiographs.

⚠️  EXPERIMENTAL — BASELINE ONLY, NOT CLINICAL.

When no tooth segmentation model or tooth boxes are available, this module
provides a simple heuristic to propose approximate tooth bounding boxes
in bitewing radiographs.

Method:
    1. Convert to grayscale and enhance contrast (CLAHE).
    2. Apply adaptive thresholding to segment bright tooth regions.
    3. Use vertical projection profile to find tooth column boundaries.
    4. Use horizontal projection to find jaw row boundaries.
    5. Combine to produce per-tooth bounding box proposals.

These proposals are intentionally rough and should be replaced by a
trained tooth detector for production use.
"""

import logging
from typing import List, Tuple

import numpy as np

from .tooth_instance import ToothInstance

log = logging.getLogger(__name__)


def propose_teeth_heuristic(
    image: np.ndarray,
    min_tooth_width: int = 30,
    max_tooth_width: int = 200,
    min_tooth_height: int = 40,
    max_tooth_height: int = 300,
) -> List[ToothInstance]:
    """
    Generate heuristic tooth bounding box proposals from a bitewing image.

    ⚠️ EXPERIMENTAL BASELINE — not suitable for clinical use.

    Args:
        image: Input image as numpy array (BGR or grayscale).
        min_tooth_width: Minimum expected tooth width in pixels.
        max_tooth_width: Maximum expected tooth width in pixels.
        min_tooth_height: Minimum expected tooth height in pixels.
        max_tooth_height: Maximum expected tooth height in pixels.

    Returns:
        List of ToothInstance with proposed bounding boxes.
    """
    import cv2

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    # Enhance contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Adaptive threshold to segment bright regions (teeth)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=51, C=-10,
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # ── Vertical projection → find tooth column boundaries ──
    v_proj = np.sum(binary, axis=0) / 255.0  # sum of white pixels per column
    v_threshold = np.max(v_proj) * 0.15

    # Find runs of columns above threshold
    above = v_proj > v_threshold
    columns = _find_runs(above, min_length=min_tooth_width)

    # ── Horizontal projection → find jaw row boundaries ──
    h_proj = np.sum(binary, axis=1) / 255.0
    h_threshold = np.max(h_proj) * 0.15
    rows_above = h_proj > h_threshold
    rows = _find_runs(rows_above, min_length=min_tooth_height)

    if not rows:
        # Fallback: use full image height with some margin
        rows = [(int(h * 0.1), int(h * 0.9))]

    if not columns:
        log.warning("Heuristic found no tooth columns — returning empty proposals.")
        return []

    # Combine rows and columns to form tooth boxes
    teeth = []
    tooth_id = 0
    for row_start, row_end in rows:
        row_h = row_end - row_start
        if row_h < min_tooth_height or row_h > max_tooth_height * 2:
            continue
        for col_start, col_end in columns:
            col_w = col_end - col_start
            if col_w < min_tooth_width or col_w > max_tooth_width:
                continue
            teeth.append(
                ToothInstance(
                    tooth_id=tooth_id,
                    bbox=(float(col_start), float(row_start),
                          float(col_end), float(row_end)),
                    confidence=0.3,  # Low confidence = heuristic
                )
            )
            tooth_id += 1

    log.info(f"Heuristic proposed {len(teeth)} tooth boxes "
             f"(rows={len(rows)}, columns={len(columns)})")
    return teeth


def _find_runs(mask: np.ndarray, min_length: int) -> List[Tuple[int, int]]:
    """
    Find contiguous runs of True values in a 1D boolean array.

    Returns list of (start, end) index tuples for runs >= min_length.
    """
    runs = []
    in_run = False
    start = 0

    for i, val in enumerate(mask):
        if val and not in_run:
            start = i
            in_run = True
        elif not val and in_run:
            if i - start >= min_length:
                runs.append((start, i))
            in_run = False

    if in_run and len(mask) - start >= min_length:
        runs.append((start, len(mask)))

    return runs


# ── Self-test ──
if __name__ == "__main__":
    print("Testing heuristic tooth proposals ...")
    # Create a synthetic bitewing-like image
    img = np.zeros((400, 600), dtype=np.uint8)
    # Draw 4 "teeth" as bright rectangles
    for i, x in enumerate([50, 170, 290, 410]):
        img[100:300, x:x + 80] = 200 + i * 10

    teeth = propose_teeth_heuristic(img)
    print(f"✓ Proposed {len(teeth)} teeth from synthetic image.")
    for t in teeth:
        print(f"  Tooth {t.tooth_id}: bbox={t.bbox}")
