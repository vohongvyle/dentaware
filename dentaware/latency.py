"""Step 5: efficiency. Time per radiograph for tooth segmentation and disease detection."""

from __future__ import annotations

import time

import numpy as np

from .config import Config
from .io_utils import read_gray, save_json
from .paths import build_index


def run_latency(cfg: Config, variant: str = "baseline", n_images: int = 10, log=print) -> dict:
    import torch

    from .seg.experiments import load_tooth_model
    from .seg.model import predict_prob

    device = cfg.resolve_device()
    idx = build_index(cfg.data_root)
    imgs = [read_gray(s.image) for s in [s for s in idx.child_det if s.split == "test"][:n_images]]
    res = {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}

    seg_path = cfg.out / "seg" / "tooth_unet.pt"
    if seg_path.exists():
        model, size = load_tooth_model(seg_path, device)
        predict_prob(model, imgs[0], size=size, device=device)  # warm-up
        t = []
        for im in imgs:
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            predict_prob(model, im, size=size, device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            t.append(time.perf_counter() - t0)
        res["segmentation_ms"] = float(np.median(t) * 1000)
        res["segmentation_params_millions"] = sum(p.numel() for p in model.parameters()) / 1e6

    w = cfg.out / "det" / variant / "final" / "train" / "weights" / "best.pt"
    inputs = cfg.out / "det" / variant / "inputs" / "test"
    if w.exists() and inputs.exists():
        from ultralytics import YOLO
        yolo = YOLO(str(w))
        files = sorted(inputs.glob("*.png"))[:n_images]
        dev = 0 if device == "cuda" else device
        yolo.predict(str(files[0]), imgsz=cfg.det_img_size, device=dev, verbose=False)
        t = []
        for f in files:
            t0 = time.perf_counter()
            yolo.predict(str(f), imgsz=cfg.det_img_size, device=dev, verbose=False)
            t.append(time.perf_counter() - t0)
        res["detection_ms"] = float(np.median(t) * 1000)
        res["detection_params_millions"] = sum(p.numel() for p in yolo.model.parameters()) / 1e6
    save_json(res, cfg.out / "latency.json")
    log(f"Latency: {res}")
    return res
