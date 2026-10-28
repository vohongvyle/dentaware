"""Figures for the report: ground truth vs. predictions with tooth assignment."""

from __future__ import annotations

import numpy as np

from .config import Config
from .io_utils import load_json, read_gray
from .labels import CLASSES
from .teeth import assign_lesions, split_teeth

PALETTE = ["#e6550d", "#3182bd", "#31a354", "#756bb1", "#de2d26", "#636363"]


def figure_examples(cfg: Config, variant: str, n: int = 4, threshold: float = 0.4, split: str = "test") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    from .det.yolo_data import tooth_prob_path
    from .paths import build_index

    recs = load_json(cfg.out / "det" / "records" / f"{variant}_{split}.json")
    recs = sorted(recs, key=lambda r: -len(r["gt"]))[:n]
    by_id = {s.id: s for s in build_index(cfg.data_root).child_det}
    fig, axes = plt.subplots(len(recs), 1, figsize=(10, 3.3 * len(recs)))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, recs):
        s = by_id[r["image_id"]]
        img = read_gray(s.image)
        ax.imshow(img, cmap="gray")
        tp = tooth_prob_path(cfg.out, s)
        names = [None] * len(r["pred"])
        if tp.exists():
            labels, teeth = split_teeth((read_gray(tp) > 127).astype(np.uint8))
            ax.contour(labels > 0, levels=[0.5], colors=["#9ecae1"], linewidths=0.6)
            names = [a["tooth"] for a in assign_lesions(r["pred"], labels, teeth)]
        for g in r["gt"]:
            ax.add_patch(mpatches.Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False,
                                            edgecolor="white", linestyle="--", linewidth=1))
        for p, name in zip(r["pred"], names):
            if p[4] < threshold:
                continue
            c = PALETTE[int(p[5]) % len(PALETTE)]
            ax.add_patch(mpatches.Rectangle((p[0], p[1]), p[2] - p[0], p[3] - p[1], fill=False, edgecolor=c, linewidth=1.4))
            ax.text(p[0], p[1] - 3, f"{CLASSES[int(p[5])]} {p[4]:.2f}" + (f" · {name}" if name else ""),
                    color=c, fontsize=7, va="bottom")
        ax.set_title(f"{s.image.name}  (white dashed = expert label)", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    path = cfg.out / "figures" / f"examples_{variant}_{split}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def figure_segmentation_comparison(cfg: Config) -> str | None:
    """Test Dice per segmentation strategy with 95% CIs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = cfg.out / "seg" / "seg_summary.json"
    if not p.exists():
        return None
    res = load_json(p)["results"]
    names = list(res)
    m = [res[k]["test_dice"] for k in names]
    lo = [res[k]["test_dice"] - res[k]["test_dice_ci"][0] for k in names]
    hi = [res[k]["test_dice_ci"][1] - res[k]["test_dice"] for k in names]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.errorbar(range(len(names)), m, yerr=[lo, hi], fmt="o", capsize=4)
    ax.set_xticks(range(len(names)), [n.replace("_", " ") for n in names])
    ax.set_ylabel("test Dice (95% CI)")
    ax.set_title("Pediatric tooth segmentation")
    fig.tight_layout()
    out = cfg.out / "figures" / "segmentation_dice.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def figure_detection_comparison(cfg: Config) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    p = cfg.out / "det" / "det_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[df["variant"].isin(["baseline", "fused", "filtered"])]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for j, split in enumerate(["oof", "test"]):
        d = df[df["split"] == split]
        x = np.arange(len(d)) + (j - 0.5) * 0.25
        ax.errorbar(x, d["mAP50"], yerr=[d["mAP50"] - d["ci_low"], d["ci_high"] - d["mAP50"]], fmt="o", capsize=4,
                    label="cross-validation" if split == "oof" else "test")
        ax.set_xticks(np.arange(len(d)), d["variant"])
    ax.set_ylabel("mAP@0.5 (95% CI)")
    ax.set_title("Disease detection")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = cfg.out / "figures" / "detection_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)
