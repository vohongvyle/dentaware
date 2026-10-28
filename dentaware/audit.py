"""Step 1: audit the data before training anything.

Writes runs/<name>/audit/ with
  summary.json        subset sizes, mask encodings, label problems
  boxes.csv           one row per disease box
  class_counts.csv    boxes per class and split
  box_sizes.png       lesion size distribution per class
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .config import Config
from .io_utils import read_gray, read_mask, save_json
from .labels import CLASSES, parse_labelme
from .paths import build_index, match_det_to_child_masks


def run_audit(cfg: Config) -> dict:
    out = cfg.out / "audit"
    out.mkdir(parents=True, exist_ok=True)
    idx = build_index(cfg.data_root)

    # Mask encodings: how many distinct raw encodings exist per subset?
    enc = Counter()
    for s in idx.child_seg + idx.adult_seg[:300]:
        _, info = read_mask(s.mask)
        enc[(s.subset, s.split, info["max"])] += 1
    encodings = [{"subset": a, "split": b, "raw_max_value": c, "n": n} for (a, b, c), n in sorted(enc.items())]

    rows, problems = [], {"unknown_labels": Counter(), "clipped": 0, "degenerate": 0, "images_without_boxes": 0}
    sizes = []
    for s in idx.child_det:
        h, w = read_gray(s.image).shape
        sizes.append((w, h))
        boxes, rep = parse_labelme(s.label, w, h)
        problems["unknown_labels"].update(rep["unknown_labels"])
        problems["clipped"] += rep["clipped"]
        problems["degenerate"] += rep["degenerate"]
        if not boxes:
            problems["images_without_boxes"] += 1
        for b in boxes:
            rows.append({"image_id": s.id, "split": s.split, "cls": CLASSES[b.cls],
                         "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                         "w": b.x2 - b.x1, "h": b.y2 - b.y1, "area_frac": b.area / (w * h)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "boxes.csv", index=False)
    counts = (df.groupby(["cls", "split"]).size().unstack(fill_value=0).reindex(CLASSES, fill_value=0)
              if len(df) else pd.DataFrame())
    counts.to_csv(out / "class_counts.csv")

    oracle = match_det_to_child_masks(idx)
    summary = {
        "index": idx.summary(),
        "mask_encodings": encodings,
        "detection": {
            "n_images": len(idx.child_det),
            "n_boxes": len(df),
            "boxes_per_class": counts.sum(axis=1).to_dict() if len(df) else {},
            "image_sizes": {f"{w}x{h}": n for (w, h), n in Counter(sizes).most_common(5)},
            "median_box_area_frac": float(df["area_frac"].median()) if len(df) else None,
            "unknown_labels": dict(problems["unknown_labels"]),
            "clipped_boxes": problems["clipped"],
            "degenerate_boxes_dropped": problems["degenerate"],
            "images_without_boxes": problems["images_without_boxes"],
        },
        "detection_images_with_true_tooth_mask": len(oracle),
    }
    save_json(summary, out / "summary.json")
    _plot_sizes(df, out / "box_sizes.png")
    return summary


def _plot_sizes(df: pd.DataFrame, path) -> None:
    if not len(df):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.5))
    present = [c for c in CLASSES if c in set(df["cls"])]
    data = [np.sqrt(df.loc[df["cls"] == c, "area_frac"]) * 100 for c in present]
    ax.boxplot(data, vert=False)
    ax.set_yticks(range(1, len(present) + 1), present)
    ax.set_xlabel("lesion size (sqrt of box area, % of image side)")
    ax.set_title("Lesion size by class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
