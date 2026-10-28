"""Step 4 (RQ4): can the detector tell when it is unsure?

* Calibration: expected calibration error (ECE) of detection confidences, before and after
  isotonic recalibration. Recalibration is cross-fitted on the out-of-fold predictions (each fold
  calibrated by a mapping fitted on the other folds) and, for the test split, fitted on all OOF.
* Triage: images whose detections all have confident scores are 'handled automatically'; images
  with any uncertain detection are 'referred to a dentist'. We report image-level F1 on the
  handled images against the share referred, for a range of uncertainty bands.
* Lesion-to-tooth table: every confident test detection with the tooth it sits on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .config import Config
from .det.evaluate import detection_outcomes
from .io_utils import load_json, read_gray, save_json
from .labels import CLASSES
from .teeth import assign_lesions, split_teeth


def ece(scores: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    if scores.size == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (scores > lo) & (scores <= hi) if lo > 0 else (scores >= lo) & (scores <= hi)
        if m.any():
            total += m.mean() * abs(scores[m].mean() - correct[m].mean())
    return float(total)


def _outcomes_df(records, min_score) -> pd.DataFrame:
    return pd.DataFrame(detection_outcomes(records, min_score=min_score))


def calibrate(oof_records, test_records, folds, cfg: Config, min_score: float = 0.05) -> dict:
    oof = _outcomes_df(oof_records, min_score)
    test = _outcomes_df(test_records, min_score)
    if oof.empty:
        return {}
    fold_of = {i: k for k, ids in enumerate(folds) for i in ids}
    oof["fold"] = oof["image_id"].map(fold_of)
    oof["calibrated"] = np.nan
    for k in sorted(oof["fold"].dropna().unique()):
        tr, te = oof["fold"] != k, oof["fold"] == k
        if tr.sum() >= 5 and oof.loc[tr, "correct"].nunique() > 1:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof.loc[tr, "score"], oof.loc[tr, "correct"])
            oof.loc[te, "calibrated"] = iso.predict(oof.loc[te, "score"])
        else:
            oof.loc[te, "calibrated"] = oof.loc[te, "score"]
    full = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof["score"], oof["correct"]) \
        if oof["correct"].nunique() > 1 else None
    if not test.empty:
        test["calibrated"] = full.predict(test["score"]) if full is not None else test["score"]
    res = {}
    for name, df in (("oof", oof), ("test", test)):
        if df.empty:
            continue
        res[name] = {"n_detections": int(len(df)), "precision": float(df["correct"].mean()),
                     "ece_raw": ece(df["score"].to_numpy(), df["correct"].to_numpy(), cfg.ece_bins),
                     "ece_calibrated": ece(df["calibrated"].to_numpy(), df["correct"].to_numpy(), cfg.ece_bins)}
    return {"metrics": res, "oof": oof, "test": test, "mapper": full}


def reliability_plot(df: pd.DataFrame, path, bins: int = 10, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = np.linspace(0, 1, bins + 1)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="0.5", label="perfect calibration")
    for col, lab in (("score", "raw confidence"), ("calibrated", "after isotonic")):
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (df[col] > lo) & (df[col] <= hi)
            if m.sum() >= 3:
                xs.append(df.loc[m, col].mean()); ys.append(df.loc[m, "correct"].mean())
        ax.plot(xs, ys, marker="o", label=lab)
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("observed precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title or "Reliability diagram")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def triage_curve(records, mapper, op_threshold: float, bands=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)) -> pd.DataFrame:
    """Refer an image if any detection's calibrated score falls within +/- band of the operating threshold."""
    rows = []
    for band in bands:
        handled = tp = fp = fn = 0
        for r in records:
            preds = r["pred"]
            scores = mapper.predict([p[4] for p in preds]) if (mapper is not None and preds) else np.array([p[4] for p in preds])
            if band > 0 and np.any(np.abs(scores - op_threshold) < band):
                continue
            handled += 1
            keep = [{"image_id": r["image_id"], "width": r["width"], "height": r["height"], "gt": r["gt"],
                     "pred": [p for p, s in zip(preds, scores) if s >= op_threshold]}]
            outs = detection_outcomes(keep)
            t = sum(o["correct"] for o in outs)
            tp += t; fp += len(outs) - t; fn += len(r["gt"]) - t
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
        rows.append({"band": band, "share_referred": 1 - handled / max(len(records), 1), "f1_on_handled": f1,
                     "n_handled": handled})
    return pd.DataFrame(rows)


def best_f1_threshold(records, mapper) -> float:
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        curve = triage_curve(records, mapper, float(t), bands=(0.0,))
        f = curve["f1_on_handled"].iloc[0]
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def run_reliability(cfg: Config, variant: str | None = None, log=print) -> dict:
    det = cfg.out / "det"
    summary = load_json(det / "det_summary.json")
    oof_rows = [r for r in summary["table"] if r["split"] == "oof" and r["variant"] in ("baseline", "fused", "filtered")]
    variant = variant or max(oof_rows, key=lambda r: r["mAP50"])["variant"]
    oof_rec = load_json(det / "records" / f"{variant}_oof.json")
    test_rec = load_json(det / "records" / f"{variant}_test.json")
    folds = load_json(det / "folds.json")["folds"]
    out = cfg.out / "reliability"
    out.mkdir(parents=True, exist_ok=True)
    log(f"Reliability analysis on variant '{variant}' (best OOF mAP50)")

    cal = calibrate(oof_rec, test_rec, folds, cfg)
    metrics = cal.get("metrics", {})
    for name in ("oof", "test"):
        if name in cal and not cal[name].empty:
            reliability_plot(cal[name], out / f"reliability_{name}.png", cfg.ece_bins, f"{variant} ({name})")
            cal[name].to_csv(out / f"detections_{name}.csv", index=False)

    op = best_f1_threshold(oof_rec, cal.get("mapper"))
    tri_oof = triage_curve(oof_rec, cal.get("mapper"), op)
    tri_test = triage_curve(test_rec, cal.get("mapper"), op)
    tri_oof.to_csv(out / "triage_oof.csv", index=False)
    tri_test.to_csv(out / "triage_test.csv", index=False)
    _triage_plot(tri_oof, tri_test, out / "triage.png")

    lesions = lesion_table(cfg, test_rec, cal.get("mapper"), op)
    lesions.to_csv(out / "lesion_teeth_test.csv", index=False)
    res = {"variant": variant, "operating_threshold_from_oof": op, "calibration": metrics,
           "lesions_assigned_to_a_tooth": float(lesions["tooth"].notna().mean()) if len(lesions) else None}
    save_json(res, out / "reliability_summary.json")
    log(f"ECE: {metrics}")
    return res


def lesion_table(cfg: Config, records, mapper, threshold: float) -> pd.DataFrame:
    from .det.yolo_data import tooth_prob_path
    from .paths import build_index
    idx = build_index(cfg.data_root)
    by_id = {s.id: s for s in idx.child_det}
    rows = []
    for r in records:
        s = by_id[r["image_id"]]
        p = tooth_prob_path(cfg.out, s)
        if not p.exists() or not r["pred"]:
            continue
        labels, teeth = split_teeth((read_gray(p) > 127).astype(np.uint8))
        scores = mapper.predict([q[4] for q in r["pred"]]) if mapper is not None else [q[4] for q in r["pred"]]
        kept = [(q, sc) for q, sc in zip(r["pred"], scores) if sc >= threshold]
        for (q, sc), a in zip(kept, assign_lesions([q for q, _ in kept], labels, teeth)):
            rows.append({"image_id": r["image_id"], "disease": CLASSES[int(q[5])], "confidence": float(sc),
                         "tooth": a["tooth"], "tooth_overlap": a["overlap"], "x1": q[0], "y1": q[1], "x2": q[2], "y2": q[3]})
    return pd.DataFrame(rows)


def _triage_plot(oof: pd.DataFrame, test: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for df, lab in ((oof, "cross-validation"), (test, "test")):
        ax.plot(df["share_referred"] * 100, df["f1_on_handled"], marker="o", label=lab)
    ax.set_xlabel("images referred to a dentist (%)")
    ax.set_ylabel("F1 on images handled automatically")
    ax.set_title("Accuracy vs. referral rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
