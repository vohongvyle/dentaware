"""Detector backends.

YoloBackend wraps Ultralytics. FakeBackend returns noisy copies of the ground truth; it exists
only so the whole pipeline (folds, filtering, evaluation, calibration, plots) can be smoke-tested
on a machine without a GPU or PyTorch. Never report numbers from FakeBackend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import Config


class YoloBackend:
    name = "yolo"

    def train(self, data_yaml: Path, run_dir: Path, cfg: Config) -> Path:
        from ultralytics import YOLO
        model = YOLO(cfg.det_model)
        model.train(
            data=str(data_yaml), imgsz=cfg.det_img_size, epochs=cfg.det_epochs, patience=cfg.det_patience,
            batch=cfg.det_batch, project=str(run_dir.parent), name=run_dir.name, exist_ok=True, seed=cfg.seed,
            deterministic=True, device=_device(cfg), verbose=False, plots=False,
            # Radiographs are grayscale and the 'fused' variant stores tooth probability in a colour
            # channel, so colour augmentations are switched off for every variant (fair comparison).
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=10,
        )
        best = run_dir / "weights" / "best.pt"
        return best if best.exists() else run_dir / "weights" / "last.pt"

    def predict(self, weights: Path, images: dict[str, Path], cfg: Config) -> dict[str, list[list[float]]]:
        from ultralytics import YOLO
        model = YOLO(str(weights))
        out = {}
        for sid, path in images.items():
            r = model.predict(source=str(path), imgsz=cfg.det_img_size, conf=cfg.det_conf_floor, iou=0.6,
                              max_det=100, device=_device(cfg), verbose=False)[0]
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4))
            conf = b.conf.cpu().numpy() if len(b) else np.zeros(0)
            cls = b.cls.cpu().numpy() if len(b) else np.zeros(0)
            out[sid] = [[*map(float, xyxy[i]), float(conf[i]), int(cls[i])] for i in range(len(conf))]
        return out


def _device(cfg: Config):
    d = cfg.resolve_device()
    return 0 if d == "cuda" else d


class FakeBackend:
    """Smoke-test stand-in: jittered ground truth plus random false positives."""
    name = "fake"

    def __init__(self, gts: dict, seed: int = 0, quality: dict | None = None):
        self.gts = gts
        self.rng = np.random.default_rng(seed)
        self.quality = quality or {"baseline": 0.6, "fused": 0.7}
        self.variant = "baseline"

    def train(self, data_yaml: Path, run_dir: Path, cfg: Config) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.variant = "fused" if "/fused/" in str(run_dir).replace("\\", "/") else "baseline"
        p = run_dir / "fake.pt"
        p.write_text(self.variant)
        return p

    def predict(self, weights: Path, images: dict[str, Path], cfg: Config) -> dict[str, list[list[float]]]:
        q = self.quality.get(Path(weights).read_text(), 0.6)
        out = {}
        for sid in images:
            preds = []
            for b in self.gts[sid]:
                if self.rng.random() < q + 0.25:
                    j = self.rng.normal(0, 2, 4)
                    preds.append([b.x1 + j[0], b.y1 + j[1], b.x2 + j[2], b.y2 + j[3],
                                  float(np.clip(self.rng.normal(q, 0.2), 0.01, 0.99)), b.cls])
            for _ in range(int(self.rng.integers(0, 4))):
                x, y = self.rng.uniform(0, 400), self.rng.uniform(0, 200)
                preds.append([x, y, x + 20, y + 30, float(self.rng.uniform(0.01, 0.6)), int(self.rng.integers(0, 6))])
            out[sid] = preds
        return out
