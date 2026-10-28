"""Step 3 (RQ1, RQ2): disease detection with cross-validation and tooth awareness.

Protocol
  * The official 70-image training split is divided into K stratified folds. Each fold is predicted
    by a model trained on the other folds, giving out-of-fold (OOF) predictions for every training
    image. All choices (variant ranking, filter threshold, calibration) use OOF predictions only.
  * A final model per variant is trained on the whole training split (with an internal validation
    split for early stopping) and evaluated once on the official 30-image test split.

Variants
  baseline  radiograph only
  fused     radiograph + predicted tooth map as an input channel
  filtered  baseline detections whose box lies mostly off the predicted teeth are removed
            (threshold chosen on OOF predictions)
  oracle_filtered  as 'filtered' but with ground-truth tooth masks, on the images where they exist
                   (an upper bound for what better tooth masks could give)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..io_utils import read_gray, read_mask, save_json, seed_everything
from ..labels import CLASSES, parse_labelme
from ..paths import build_index, match_det_to_child_masks
from ..splits import stratified_folds, train_val_split
from ..stats import bootstrap_statistic
from ..teeth import tooth_overlap
from .evaluate import evaluate_records, map_stat, paired_bootstrap_map, size_buckets
from .yolo_data import render_inputs, tooth_prob_path, write_dataset

FILTER_THRESHOLDS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)


def _records(samples, gts, sizes, preds) -> list[dict]:
    recs = []
    for s in samples:
        w, h = sizes[s.id]
        recs.append({"image_id": s.id, "width": w, "height": h,
                     "gt": [[b.x1, b.y1, b.x2, b.y2, b.cls] for b in gts[s.id]],
                     "pred": preds.get(s.id, [])})
    return recs


def _filter_records(recs: list[dict], masks: dict[str, np.ndarray], tau: float) -> list[dict]:
    out = []
    for r in recs:
        m = masks.get(r["image_id"])
        if m is None:
            out.append(r)
            continue
        keep = [p for p in r["pred"] if tooth_overlap(p, m) >= tau]
        out.append({**r, "pred": keep})
    return out


def _train_or_resume(backend, data_yaml, run_dir, cfg: Config, log):
    """Skip training when finished weights already exist (lets a long run resume after Colab disconnects)."""
    done = run_dir / "weights" / "best.pt"
    if getattr(backend, "name", "") == "yolo" and done.exists() and (run_dir / "results.csv").exists():
        log(f"  reusing {done}")
        return done
    return backend.train(data_yaml, run_dir, cfg)


def run_detection(cfg: Config, backend=None, log=print) -> dict:
    seed_everything(cfg.seed)
    out = cfg.out / "det"
    out.mkdir(parents=True, exist_ok=True)
    idx = build_index(cfg.data_root)
    train_s = [s for s in idx.child_det if s.split == "train"]
    test_s = [s for s in idx.child_det if s.split == "test"]

    gts, sizes = {}, {}
    for s in train_s + test_s:
        h, w = read_gray(s.image).shape
        sizes[s.id] = (w, h)
        gts[s.id], _ = parse_labelme(s.label, w, h)

    if backend is None:
        from .backend import YoloBackend
        backend = YoloBackend()

    counts = [np.bincount([b.cls for b in gts[s.id]], minlength=len(CLASSES)) for s in train_s]
    folds = stratified_folds([s.id for s in train_s], counts, cfg.det_folds, cfg.seed)
    save_json({"folds": folds}, out / "folds.json")
    by_id = {s.id: s for s in train_s}
    log(f"Detection: {len(train_s)} train images in {cfg.det_folds} folds, {len(test_s)} test images")

    all_records: dict[str, dict[str, list[dict]]] = {}
    for variant in cfg.det_variants:
        if variant == "fused" and not all(tooth_prob_path(cfg.out, s).exists() for s in train_s + test_s):
            log("Skipping 'fused': tooth masks are missing (run the segmentation step first).")
            continue
        log(f"\n=== detection variant: {variant} ===")
        inputs = render_inputs(train_s + test_s, variant, cfg.out)
        oof_preds = {}
        for k, fold in enumerate(folds):
            val = [by_id[i] for i in fold]
            tr = [by_id[i] for f, ids in enumerate(folds) if f != k for i in ids]
            ds = write_dataset(tr, val, gts, sizes, inputs, out / variant / f"fold{k}" / "data")
            w = _train_or_resume(backend, ds, out / variant / f"fold{k}" / "train", cfg, log)
            oof_preds.update(backend.predict(w, {s.id: inputs[s.id] for s in val}, cfg))
            log(f"  fold {k + 1}/{len(folds)} done")
        tr_ids, es_ids = train_val_split([s.id for s in train_s], 0.15, cfg.seed)
        ds = write_dataset([by_id[i] for i in tr_ids], [by_id[i] for i in es_ids], gts, sizes, inputs,
                           out / variant / "final" / "data")
        w = _train_or_resume(backend, ds, out / variant / "final" / "train", cfg, log)
        test_preds = backend.predict(w, {s.id: inputs[s.id] for s in test_s}, cfg)
        all_records[variant] = {"oof": _records(train_s, gts, sizes, oof_preds),
                                "test": _records(test_s, gts, sizes, test_preds)}

    # Post-hoc tooth filtering of the baseline (threshold chosen on OOF only)
    if "baseline" in all_records:
        pred_masks = {}
        for s in train_s + test_s:
            p = tooth_prob_path(cfg.out, s)
            if p.exists():
                pred_masks[s.id] = (read_gray(p) > 127).astype(np.uint8)
        if pred_masks:
            scores = {t: map_stat()(_filter_records(all_records["baseline"]["oof"], pred_masks, t)) for t in FILTER_THRESHOLDS}
            tau = max(scores, key=lambda t: (scores[t], -t))
            all_records["filtered"] = {k: _filter_records(v, pred_masks, tau) for k, v in all_records["baseline"].items()}
            save_json({"oof_map_by_threshold": scores, "chosen_threshold": tau}, out / "filter_threshold.json")
            log(f"Tooth filter threshold chosen on OOF: {tau} (OOF mAP by threshold: {scores})")
        oracle = match_det_to_child_masks(idx)
        if oracle:
            gt_masks = {sid: read_mask(p)[0] for sid, p in oracle.items()}
            tau_o = 0.1
            keep = set(gt_masks)
            all_records["oracle_filtered"] = {k: _filter_records([r for r in v if r["image_id"] in keep], gt_masks, tau_o)
                                              for k, v in all_records["baseline"].items()}
            all_records["baseline_oracle_subset"] = {k: [r for r in v if r["image_id"] in keep]
                                                     for k, v in all_records["baseline"].items()}

    for variant, recs in all_records.items():
        for split, r in recs.items():
            save_json(r, out / "records" / f"{variant}_{split}.json")

    summary = summarise(all_records, cfg, log)
    save_json(summary, out / "det_summary.json")
    return summary


def summarise(all_records: dict, cfg: Config, log=print) -> dict:
    rows, comparisons = [], {}
    ref = all_records.get("baseline", {}).get("oof", [])
    buckets = size_buckets(ref) if ref else {}
    for variant, recs in all_records.items():
        for split, r in recs.items():
            if not r:
                continue
            m, lo, hi = bootstrap_statistic(r, map_stat(cfg.det_iou_match), cfg.n_bootstrap, cfg.seed)
            per_class = evaluate_records(r, cfg.det_iou_match)["per_class"]
            row = {"variant": variant, "split": split, "n_images": len(r), "mAP50": m, "ci_low": lo, "ci_high": hi,
                   **{f"AP50_{c}": per_class[c] for c in CLASSES}}
            for name, rng in buckets.items():
                row[f"mAP50_{name}"] = evaluate_records(r, cfg.det_iou_match, rng)["mAP"]
            rows.append(row)
            log(f"{variant:>24s} {split:>4s}  mAP50 {m:.3f} [{lo:.3f}, {hi:.3f}]  n={len(r)}")
    base = all_records.get("baseline")
    for variant, recs in all_records.items():
        if base is None or variant in ("baseline", "baseline_oracle_subset"):
            continue
        ref_recs = all_records["baseline_oracle_subset"] if variant == "oracle_filtered" else base
        for split in recs:
            if recs[split] and ref_recs.get(split):
                comparisons[f"{variant}_vs_baseline_{split}"] = paired_bootstrap_map(
                    ref_recs[split], recs[split], cfg.n_bootstrap, cfg.seed, cfg.det_iou_match)
                if "small" in buckets:
                    comparisons[f"{variant}_vs_baseline_{split}_small"] = paired_bootstrap_map(
                        ref_recs[split], recs[split], cfg.n_bootstrap, cfg.seed, cfg.det_iou_match, buckets["small"])
    table = pd.DataFrame(rows)
    table.to_csv(cfg.out / "det" / "det_results.csv", index=False)
    return {"table": rows, "comparisons": comparisons, "size_buckets": buckets}
