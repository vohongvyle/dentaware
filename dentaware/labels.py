"""Disease labels: LabelMe JSON parsing and the class list.

The detection subset is annotated in Chinese. The English names follow Zhang et al. (2023).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io_utils import load_json

LABEL_MAP = {
    "龋病": "caries",
    "根尖周炎": "periapical_lesion",
    "牙髓炎": "pulpitis",
    "深窝沟": "deep_fissure",
    "牙齿发育异常": "developmental_anomaly",
    "其他": "other",
}
CLASSES = ["caries", "periapical_lesion", "pulpitis", "deep_fissure", "developmental_anomaly", "other"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int
    score: float = 1.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


def normalise_label(raw: str) -> str | None:
    raw = (raw or "").strip()
    if raw in LABEL_MAP:
        return LABEL_MAP[raw]
    low = raw.lower().replace(" ", "_")
    if low in CLASS_TO_ID:
        return low
    return None


def parse_labelme(path: str | Path, img_w: int | None = None, img_h: int | None = None) -> tuple[list[Box], dict]:
    """Return boxes and a report of anything unusual (unknown labels, clipped or degenerate boxes)."""
    data = load_json(path)
    W = img_w or data.get("imageWidth")
    H = img_h or data.get("imageHeight")
    boxes: list[Box] = []
    report = {"unknown_labels": [], "clipped": 0, "degenerate": 0, "shape_types": {}}
    for shape in data.get("shapes", []):
        st = shape.get("shape_type", "polygon")
        report["shape_types"][st] = report["shape_types"].get(st, 0) + 1
        name = normalise_label(shape.get("label", ""))
        if name is None:
            report["unknown_labels"].append(shape.get("label", ""))
            continue
        pts = np.asarray(shape.get("points", []), dtype=float)
        if pts.size < 4:
            report["degenerate"] += 1
            continue
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        if W and H:
            cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)
            if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
                report["clipped"] += 1
            x1, y1, x2, y2 = cx1, cy1, cx2, cy2
        if x2 - x1 < 2 or y2 - y1 < 2:
            report["degenerate"] += 1
            continue
        boxes.append(Box(float(x1), float(y1), float(x2), float(y2), CLASS_TO_ID[name]))
    return boxes, report


def iou(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
