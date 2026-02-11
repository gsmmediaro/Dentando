#!/usr/bin/env python3
"""
Tooth instance representation.

Dataclasses for representing individual teeth and per-tooth predictions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ToothInstance:
    """
    Represents a single tooth instance in an image.

    Attributes:
        tooth_id: Unique ID for this tooth in the image (0-indexed).
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        fdi_number: Optional FDI notation number (e.g., 48 = quadrant 4, tooth 8).
        polygon: Optional polygon vertices [(x, y), ...] for tooth contour.
        confidence: Detection confidence if from a model.
    """
    tooth_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    fdi_number: Optional[int] = None
    polygon: Optional[List[Tuple[float, float]]] = None
    confidence: float = 1.0

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(0, self.width) * max(0, self.height)


@dataclass
class ToothPrediction:
    """
    A diagnosis assigned to a specific tooth.

    Attributes:
        tooth: The tooth instance this prediction belongs to.
        diagnosis: Diagnosis label (e.g., "caries", "deep_caries").
        confidence: Confidence score of the diagnosis.
        source: How this prediction was produced ("direct" or "mapped").
        lesion_bbox: Optional bounding box of the original lesion detection
                     (only set when source="mapped").
    """
    tooth: ToothInstance
    diagnosis: str
    confidence: float = 0.0
    source: str = "direct"  # "direct" for DENTEX, "mapped" for bitewing IoU
    lesion_bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class ImagePredictions:
    """All tooth-level predictions for a single image."""
    image_path: str
    image_width: int
    image_height: int
    teeth: List[ToothInstance] = field(default_factory=list)
    predictions: List[ToothPrediction] = field(default_factory=list)
    modality: str = "unknown"  # "pano", "bitewing", "cbct"

    def affected_teeth(self) -> List[ToothPrediction]:
        """Return only teeth with a diagnosis."""
        return [p for p in self.predictions if p.diagnosis]

    def to_dict(self) -> dict:
        """Serialize to dict for JSON export."""
        return {
            "image_path": self.image_path,
            "image_size": [self.image_width, self.image_height],
            "modality": self.modality,
            "teeth": [
                {
                    "tooth_id": t.tooth_id,
                    "bbox": list(t.bbox),
                    "fdi_number": t.fdi_number,
                    "confidence": round(t.confidence, 4),
                }
                for t in self.teeth
            ],
            "predictions": [
                {
                    "tooth_id": p.tooth.tooth_id,
                    "tooth_bbox": list(p.tooth.bbox),
                    "diagnosis": p.diagnosis,
                    "confidence": round(p.confidence, 4),
                    "source": p.source,
                    "lesion_bbox": list(p.lesion_bbox) if p.lesion_bbox else None,
                }
                for p in self.predictions
            ],
        }
