"""Step 2 (RQ3): does adult data help pediatric tooth segmentation, and how should it be used?

Zhang et al. (2023) report that simply adding adult images did not improve results on children.
We compare three strategies on the *official pediatric test split*:

  child_only         train on pediatric images only
  combined           train on pediatric + adult images mixed together (the authors' setting)
  pretrain_finetune  pretrain on adult images, then fine-tune on pediatric images

Model selection uses a validation split carved from the pediatric training data. The model with
the best *validation* Dice becomes the tooth model used by the detection steps.
"""

from __future__ import annotations

import pandas as pd
import torch

from ..config import Config
from ..io_utils import read_gray, save_json, seed_everything, write_image
from ..paths import build_index
from ..splits import train_val_split
from ..stats import bootstrap_ci, paired_bootstrap_diff
from .model import evaluate, predict_prob, train_model


def run_seg_experiments(cfg: Config, log=print) -> dict:
    seed_everything(cfg.seed)
    device = cfg.resolve_device()
    out = cfg.out / "seg"
    out.mkdir(parents=True, exist_ok=True)
    idx = build_index(cfg.data_root)

    child_train = [s for s in idx.child_seg if s.split == "train"]
    child_test = [s for s in idx.child_seg if s.split == "test"]
    supplement = [s for s in idx.child_seg if s.split == "supplement"] if cfg.seg_use_supplement else []
    if not child_train or not child_test:
        raise RuntimeError("Pediatric tooth masks (train and test) are required for the segmentation experiments.")

    by_id = {s.id: s for s in child_train + supplement}
    tr_ids, va_ids = train_val_split(list(by_id), cfg.seg_val_frac, cfg.seed)
    c_train, c_val = [by_id[i] for i in tr_ids], [by_id[i] for i in va_ids]
    adult = [s for s in idx.adult_seg if "test" not in s.split]
    if cfg.seg_max_adult and len(adult) > cfg.seg_max_adult:
        import random
        adult = random.Random(cfg.seed).sample(adult, cfg.seg_max_adult)  # random subset across all 3 collections
    log(f"Pediatric train {len(c_train)} | val {len(c_val)} | test {len(child_test)} | adult {len(adult)} | device {device}")

    common = dict(size=cfg.seg_img_size, encoder=cfg.seg_encoder, lr=cfg.seg_lr, batch=cfg.seg_batch, device=device, log=log)
    results, per_image, models = {}, {}, {}
    for exp in cfg.seg_experiments:
        log(f"\n=== {exp} ===")
        seed_everything(cfg.seed)
        if exp == "child_only":
            model, info = train_model(c_train, c_val, epochs=cfg.seg_epochs, **common)
        elif exp == "combined":
            model, info = train_model(c_train + adult, c_val, epochs=cfg.seg_epochs, **common)
        elif exp == "pretrain_finetune":
            if not adult:
                log("No adult data found; skipping pretrain_finetune.")
                continue
            pre, _ = train_model(adult, c_val, epochs=cfg.seg_pretrain_epochs, **common)
            model, info = train_model(c_train, c_val, epochs=cfg.seg_epochs, init_state=pre.state_dict(),
                                      **{**common, "lr": cfg.seg_lr / 3})
        else:
            raise ValueError(exp)
        rows = evaluate(model, child_test, size=cfg.seg_img_size, device=device)
        per_image[exp] = rows
        dice = [r["dice"] for r in rows]
        iou = [r["iou"] for r in rows]
        d_m, d_lo, d_hi = bootstrap_ci(dice, cfg.n_bootstrap, seed=cfg.seed)
        i_m, i_lo, i_hi = bootstrap_ci(iou, cfg.n_bootstrap, seed=cfg.seed)
        results[exp] = {"val_dice": info["best_val_dice"], "test_dice": d_m, "test_dice_ci": [d_lo, d_hi],
                        "test_iou": i_m, "test_iou_ci": [i_lo, i_hi], "n_test": len(rows), "history": info["history"]}
        models[exp] = model
        log(f"{exp}: test Dice {d_m:.4f} [{d_lo:.4f}, {d_hi:.4f}]  IoU {i_m:.4f}")

    # Paired comparisons against the pediatric-only baseline (same test images)
    comparisons = {}
    if "child_only" in per_image:
        base = [r["dice"] for r in per_image["child_only"]]
        for exp, rows in per_image.items():
            if exp != "child_only":
                comparisons[f"{exp}_minus_child_only"] = paired_bootstrap_diff(base, [r["dice"] for r in rows],
                                                                               cfg.n_bootstrap, cfg.seed)

    best_exp = max(results, key=lambda e: results[e]["val_dice"])
    torch.save({"state_dict": models[best_exp].state_dict(), "encoder": cfg.seg_encoder, "size": cfg.seg_img_size,
                "experiment": best_exp}, out / "tooth_unet.pt")

    table = pd.DataFrame([{"experiment": e, "val_dice": r["val_dice"], "test_dice": r["test_dice"],
                           "dice_ci_low": r["test_dice_ci"][0], "dice_ci_high": r["test_dice_ci"][1],
                           "test_iou": r["test_iou"], "n_test": r["n_test"]} for e, r in results.items()])
    table.to_csv(out / "seg_results.csv", index=False)
    pd.DataFrame([{**r, "experiment": e} for e, rows in per_image.items() for r in rows]).to_csv(
        out / "seg_per_image.csv", index=False)
    summary = {"results": results, "comparisons": comparisons, "selected_by_validation": best_exp}
    save_json(summary, out / "seg_summary.json")
    log(f"\nSelected tooth model (by validation Dice): {best_exp}")

    write_tooth_masks(cfg, models[best_exp], idx, log)
    return summary


def write_tooth_masks(cfg: Config, model, idx, log=print) -> None:
    """Predict tooth probability maps for every detection image (used by the tooth-aware detector)."""
    device = cfg.resolve_device()
    out = cfg.out / "tooth_masks"
    for s in idx.child_det:
        prob = predict_prob(model, read_gray(s.image), size=cfg.seg_img_size, device=device)
        write_image(out / s.split / f"{s.image.stem}.png", (prob * 255).astype("uint8"))
    log(f"Wrote predicted tooth masks for {len(idx.child_det)} detection images to {out}")


def load_tooth_model(path, device: str):
    from .model import build_model
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["encoder"], pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["size"]
