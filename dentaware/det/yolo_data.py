"""Write detection data in YOLO format, in one of two input variants.

baseline : the radiograph in all three channels
fused    : two radiograph channels + the predicted tooth-probability map as the third channel,
           so the detector can learn where teeth are (RQ2, input-level tooth awareness)

Inputs are rendered once per variant at original resolution (so predictions map back to the
original pixels), then copied into per-fold YOLO folders.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..io_utils import read_gray, write_image
from ..labels import CLASSES, Box
from ..paths import Sample


def tooth_prob_path(out_root: Path, s: Sample) -> Path:
    return out_root / "tooth_masks" / s.split / f"{s.image.stem}.png"


def make_input(s: Sample, variant: str, out_root: Path) -> np.ndarray:
    gray = read_gray(s.image)
    if variant == "baseline":
        return np.dstack([gray, gray, gray])
    if variant == "fused":
        p = tooth_prob_path(out_root, s)
        if not p.exists():
            raise FileNotFoundError(f"Missing tooth mask {p}. Run the segmentation step first.")
        tooth = read_gray(p)
        if tooth.shape != gray.shape:
            tooth = cv2.resize(tooth, gray.shape[::-1], interpolation=cv2.INTER_LINEAR)
        return np.dstack([gray, gray, tooth])
    raise ValueError(variant)


def render_inputs(samples: list[Sample], variant: str, out_root: Path) -> dict[str, Path]:
    """Render each image once for this variant. Returns sample id -> png path."""
    folder = out_root / "det" / variant / "inputs"
    paths = {}
    for s in samples:
        p = folder / s.split / f"{s.image.stem}.png"
        if not p.exists():
            write_image(p, make_input(s, variant, out_root))
        paths[s.id] = p
    return paths


def yolo_lines(boxes: list[Box], w: int, h: int) -> list[str]:
    lines = []
    for b in boxes:
        cx, cy = (b.x1 + b.x2) / 2 / w, (b.y1 + b.y2) / 2 / h
        bw, bh = (b.x2 - b.x1) / w, (b.y2 - b.y1) / h
        lines.append(f"{b.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def write_dataset(train: list[Sample], val: list[Sample], gts: dict[str, list[Box]], sizes: dict[str, tuple[int, int]],
                  inputs: dict[str, Path], ds_dir: Path) -> Path:
    if ds_dir.exists():
        shutil.rmtree(ds_dir)
    for split, samples in (("train", train), ("val", val)):
        for s in samples:
            name = f"{s.image.stem}__{s.split}"
            dst = ds_dir / "images" / split / f"{name}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inputs[s.id], dst)
            w, h = sizes[s.id]
            lab = ds_dir / "labels" / split / f"{name}.txt"
            lab.parent.mkdir(parents=True, exist_ok=True)
            lab.write_text("\n".join(yolo_lines(gts[s.id], w, h)))
    data_yaml = ds_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(ds_dir.resolve()), "train": "images/train", "val": "images/val",
                                         "names": {i: c for i, c in enumerate(CLASSES)}}))
    return data_yaml
