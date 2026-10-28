"""All experiment settings live here, so every run is reproducible from one object."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    data_root: str = ""            # folder that contains "Dental_dataset" (or Dental_dataset itself)
    out_dir: str = "runs/main"
    seed: int = 42
    device: str = "auto"           # "auto", "cuda", "cpu"

    # ---- segmentation (tooth masks) ----
    seg_img_size: int = 512
    seg_encoder: str = "resnet34"
    seg_batch: int = 4
    seg_epochs: int = 40
    seg_pretrain_epochs: int = 15  # adult pretraining epochs for the transfer experiment
    seg_lr: float = 3e-4
    seg_val_frac: float = 0.15
    seg_use_supplement: bool = True
    seg_max_adult: int = 800       # cap adult images to fit a free Colab session (0 = all 2,692)
    seg_experiments: tuple = ("child_only", "combined", "pretrain_finetune")

    # ---- detection (diseases) ----
    det_model: str = "yolo11s.pt"
    det_img_size: int = 1024
    det_epochs: int = 80
    det_patience: int = 30
    det_batch: int = 8
    det_folds: int = 5
    det_variants: tuple = ("baseline", "fused")
    det_conf_floor: float = 0.001  # keep low-confidence boxes so AP is computed correctly
    det_iou_match: float = 0.5

    # ---- analysis ----
    n_bootstrap: int = 1000
    ece_bins: int = 10

    quick: bool = False            # tiny settings for smoke tests

    extra: dict = field(default_factory=dict)

    def apply_quick(self) -> "Config":
        if self.quick:
            self.seg_img_size, self.seg_epochs, self.seg_pretrain_epochs = 128, 2, 1
            self.seg_batch, self.seg_max_adult = 2, 8
            self.det_img_size, self.det_epochs, self.det_patience = 160, 2, 2
            self.det_batch, self.det_folds, self.n_bootstrap = 2, 2, 50
            self.det_model = self.extra.get("quick_det_model", self.det_model)
        return self

    @property
    def out(self) -> Path:
        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def save(self) -> None:
        d = asdict(self)
        (self.out / "config.json").write_text(json.dumps(d, indent=2, default=list))
