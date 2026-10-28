"""Detection evaluation written from scratch, so every number can be traced and bootstrapped.

A *record* is one image:
    {"image_id": str, "width": int, "height": int,
     "gt":   [[x1, y1, x2, y2, cls], ...],
     "pred": [[x1, y1, x2, y2, score, cls], ...]}

AP is VOC-style (all-point interpolation) at a fixed IoU threshold, per class; mAP averages the
classes that have at least one ground-truth box in the evaluated records. Size buckets follow the
COCO convention: ground truth outside the bucket is ignored, and so are unmatched predictions
outside it.
"""

from __future__ import annotations

import numpy as np

from ..labels import CLASSES, iou


def _box_area_frac(b, rec) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) / float(rec["width"] * rec["height"])


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def match_class(records: list[dict], cls: int, iou_thr: float = 0.5, area_range: tuple[float, float] | None = None):
    """Greedy matching for one class. Returns (scores, is_tp, n_gt) after applying ignore rules."""
    scores, tps, n_gt = [], [], 0
    for rec in records:
        gts = [g for g in rec["gt"] if int(g[4]) == cls]
        ignore = [area_range is not None and not (area_range[0] <= _box_area_frac(g, rec) < area_range[1]) for g in gts]
        n_gt += sum(not ig for ig in ignore)
        used = [False] * len(gts)
        preds = sorted([p for p in rec["pred"] if int(p[5]) == cls], key=lambda p: -p[4])
        for p in preds:
            best, best_j = iou_thr, -1
            for j, g in enumerate(gts):
                if used[j]:
                    continue
                o = iou(p, g)
                if o >= best:
                    best, best_j = o, j
            if best_j >= 0:
                used[best_j] = True
                if ignore[best_j]:
                    continue
                scores.append(p[4]); tps.append(1)
            else:
                if area_range is not None and not (area_range[0] <= _box_area_frac(p, rec) < area_range[1]):
                    continue
                scores.append(p[4]); tps.append(0)
    return np.asarray(scores, float), np.asarray(tps, int), n_gt


def average_precision(records: list[dict], cls: int, iou_thr: float = 0.5, area_range=None) -> float:
    scores, tps, n_gt = match_class(records, cls, iou_thr, area_range)
    if n_gt == 0:
        return float("nan")
    if tps.size == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(tps[order])
    fp = np.cumsum(1 - tps[order])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    return _voc_ap(recall, precision)


def evaluate_records(records: list[dict], iou_thr: float = 0.5, area_range=None, classes=None) -> dict:
    classes = range(len(CLASSES)) if classes is None else classes
    aps = {CLASSES[c]: average_precision(records, c, iou_thr, area_range) for c in classes}
    valid = [v for v in aps.values() if not np.isnan(v)]
    return {"mAP": float(np.mean(valid)) if valid else float("nan"), "per_class": aps}


def map_stat(iou_thr: float = 0.5, area_range=None, classes=None):
    return lambda recs: evaluate_records(recs, iou_thr, area_range, classes)["mAP"]


def size_buckets(records: list[dict]) -> dict[str, tuple[float, float]]:
    """Small / medium / large thresholds at the terciles of ground-truth box area (fraction of image)."""
    areas = [_box_area_frac(g, r) for r in records for g in r["gt"]]
    if len(areas) < 3:
        return {"all": (0.0, 1.01)}
    q1, q2 = np.quantile(areas, [1 / 3, 2 / 3])
    return {"small": (0.0, float(q1)), "medium": (float(q1), float(q2)), "large": (float(q2), 1.01)}


def paired_bootstrap_map(rec_a: list[dict], rec_b: list[dict], n: int = 1000, seed: int = 0, iou_thr: float = 0.5,
                         area_range=None) -> dict:
    """mAP(b) - mAP(a), resampling the same images for both models."""
    ida = {r["image_id"]: r for r in rec_a}
    ids = [r["image_id"] for r in rec_b if r["image_id"] in ida]
    idb = {r["image_id"]: r for r in rec_b}
    stat = map_stat(iou_thr, area_range)
    point = stat([idb[i] for i in ids]) - stat([ida[i] for i in ids])
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        s = [ids[k] for k in rng.integers(0, len(ids), len(ids))]
        d = stat([idb[i] for i in s]) - stat([ida[i] for i in s])
        if not np.isnan(d):
            diffs.append(d)
    if not diffs:
        return {"diff": point, "ci_low": float("nan"), "ci_high": float("nan"), "p_b_better": float("nan")}
    diffs = np.asarray(diffs)
    return {"diff": float(point), "ci_low": float(np.quantile(diffs, 0.025)), "ci_high": float(np.quantile(diffs, 0.975)),
            "p_b_better": float((diffs > 0).mean())}


def detection_outcomes(records: list[dict], iou_thr: float = 0.5, min_score: float = 0.0):
    """Every prediction with whether it was correct (same class, IoU >= thr, greedy by score).

    Used for calibration: a confidence of 0.7 should be correct about 70% of the time.
    """
    out = []
    for rec in records:
        for cls in {int(p[5]) for p in rec["pred"]}:
            gts = [g for g in rec["gt"] if int(g[4]) == cls]
            used = [False] * len(gts)
            for p in sorted([p for p in rec["pred"] if int(p[5]) == cls and p[4] >= min_score], key=lambda p: -p[4]):
                best, best_j = iou_thr, -1
                for j, g in enumerate(gts):
                    if not used[j]:
                        o = iou(p, g)
                        if o >= best:
                            best, best_j = o, j
                if best_j >= 0:
                    used[best_j] = True
                out.append({"image_id": rec["image_id"], "cls": cls, "score": float(p[4]), "correct": int(best_j >= 0)})
    return out
