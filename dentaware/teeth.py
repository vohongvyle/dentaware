"""Tooth-level reasoning on top of a tooth probability map.

* tooth_overlap(box, mask): share of a predicted lesion box that lies on tooth pixels
  (used to filter detections that fall on bone, soft tissue or background).
* split_teeth(mask): separate the binary tooth mask into individual teeth with a distance-transform
  watershed, then place each tooth in the upper or lower jaw and number it from the image's left.
* assign_lesions(...): attach every lesion to the tooth it overlaps most, e.g. "caries on L4".

The numbering is positional (jaw + order from image-left), not FDI numbering: the dataset has no
tooth-identity labels, so FDI numbers could not be validated.
"""

from __future__ import annotations

import cv2
import numpy as np


def tooth_overlap(box, tooth_mask: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
    h, w = tooth_mask.shape
    x1, x2 = max(0, x1), min(w, max(x2, x1 + 1))
    y1, y2 = max(0, y1), min(h, max(y2, y1 + 1))
    region = tooth_mask[y1:y2, x1:x2]
    return float(region.mean()) if region.size else 0.0


def jaw_split_row(mask: np.ndarray) -> int:
    """Row between upper and lower teeth: the least-tooth row in the central band of the image."""
    h, w = mask.shape
    band = mask[:, int(w * 0.2): int(w * 0.8)]
    profile = band.mean(axis=1)
    lo, hi = int(h * 0.3), int(h * 0.75)
    if hi <= lo:
        return h // 2
    smooth = np.convolve(profile, np.ones(9) / 9, mode="same")
    return lo + int(np.argmin(smooth[lo:hi]))


def split_teeth(mask: np.ndarray, min_area_frac: float = 0.0004) -> tuple[np.ndarray, list[dict]]:
    """Label image of tooth instances plus a list of teeth with jaw and position."""
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return np.zeros(mask.shape, np.int32), []
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    peak = dist.max()
    _, sure_fg = cv2.threshold(dist, 0.35 * peak, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    unknown = cv2.subtract(cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=2) * 255, sure_fg)
    markers[unknown == 255] = 0
    color = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers.astype(np.int32))
    labels = np.where(markers > 1, markers - 1, 0).astype(np.int32)
    labels[binary == 0] = 0

    split = jaw_split_row(binary)
    min_area = min_area_frac * mask.size
    teeth = []
    for lab in np.unique(labels):
        if lab == 0:
            continue
        ys, xs = np.nonzero(labels == lab)
        if ys.size < min_area:
            labels[labels == lab] = 0
            continue
        teeth.append({"label": int(lab), "cx": float(xs.mean()), "cy": float(ys.mean()), "area": int(ys.size),
                      "jaw": "U" if ys.mean() < split else "L"})
    for jaw in ("U", "L"):
        for k, t in enumerate(sorted((t for t in teeth if t["jaw"] == jaw), key=lambda t: t["cx"]), start=1):
            t["position"] = k
            t["name"] = f"{jaw}{k}"
    return labels, teeth


def assign_lesions(boxes: list, labels: np.ndarray, teeth: list[dict]) -> list[dict]:
    """For each box [x1,y1,x2,y2,(score),cls]: the tooth it overlaps most, and how much."""
    by_label = {t["label"]: t for t in teeth}
    out = []
    h, w = labels.shape
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
        region = labels[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        vals, counts = np.unique(region[region > 0], return_counts=True)
        if vals.size == 0:
            out.append({"tooth": None, "overlap": 0.0})
            continue
        k = int(np.argmax(counts))
        t = by_label.get(int(vals[k]))
        out.append({"tooth": t["name"] if t else None, "overlap": float(counts[k] / max(region.size, 1))})
    return out
