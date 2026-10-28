"""Image and mask reading that does not fail silently.

The original class project lost time to masks stored as 1, 128 or 255 depending on the folder,
and to paths that broke across operating systems. Everything here normalises those cases and
records what it did, so the audit can report it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_gray(path: str | Path) -> np.ndarray:
    """Read an image as uint8 grayscale. Works with non-ASCII paths (cv2.imread does not)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def read_mask(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read a binary tooth mask whatever its encoding (0/1, 0/128, 0/255, colour).

    Returns the mask as uint8 {0,1} plus a small report of the raw encoding.
    """
    raw = read_gray(path)
    values = np.unique(raw)
    vmax = int(values.max()) if values.size else 0
    mask = (raw > vmax / 2).astype(np.uint8) if vmax > 0 else np.zeros_like(raw)
    info = {"n_unique": int(values.size), "max": vmax, "values_head": [int(v) for v in values[:6]]}
    return mask, info


def write_image(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise ValueError(f"Could not encode {path}")
    buf.tofile(str(path))


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default))


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
