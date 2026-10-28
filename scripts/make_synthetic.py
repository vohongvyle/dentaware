"""Build a tiny fake dataset with the real folder layout, for smoke tests without the real data.

It deliberately includes the problems the real data has: masks encoded as 1, 128 or 255,
a label outside the class list, a box running off the image, and a degenerate box.

    python scripts/make_synthetic.py --out /tmp/fake_dental
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CHINESE = ["龋病", "根尖周炎", "牙髓炎", "深窝沟", "牙齿发育异常", "其他"]


def draw_case(rng, w=480, h=240):
    img = (rng.normal(60, 10, (h, w))).clip(0, 255).astype(np.uint8)
    mask = np.zeros((h, w), np.uint8)
    teeth = []
    for row, cy in enumerate((int(h * 0.35), int(h * 0.68))):
        n = 10
        for i in range(n):
            cx = int(w * (0.12 + 0.76 * i / (n - 1)))
            ax, ay = 14, 26
            cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1, -1)
            teeth.append((cx, cy, ax, ay))
    img[mask > 0] = np.clip(img[mask > 0].astype(int) + 120, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, mask, teeth


def write(path: Path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, arr)
    buf.tofile(str(path))


def labelme(teeth, rng, w, h, extras=True):
    shapes = []
    for _ in range(int(rng.integers(1, 5))):
        cx, cy, ax, ay = teeth[int(rng.integers(len(teeth)))]
        cls = int(rng.integers(0, 4)) if rng.random() < 0.85 else int(rng.integers(4, 6))
        dx, dy = rng.integers(4, ax), rng.integers(6, ay)
        x1, y1 = cx - dx, cy - dy
        shapes.append({"label": CHINESE[cls], "points": [[float(x1), float(y1)], [float(x1 + 2 * dx), float(y1 + 2 * dy)]],
                       "shape_type": "rectangle"})
    if extras and rng.random() < 0.3:
        shapes.append({"label": "未知", "points": [[10.0, 10.0], [30.0, 30.0]], "shape_type": "rectangle"})
    if extras and rng.random() < 0.2:
        shapes.append({"label": CHINESE[0], "points": [[w - 10.0, 20.0], [w + 15.0, 40.0]], "shape_type": "rectangle"})
    if extras and rng.random() < 0.1:
        shapes.append({"label": CHINESE[1], "points": [[50.0, 50.0], [50.5, 50.5]], "shape_type": "rectangle"})
    return {"version": "5.0.1", "shapes": shapes, "imageWidth": w, "imageHeight": h}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-det-train", type=int, default=14)
    ap.add_argument("--n-det-test", type=int, default=6)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    root = Path(args.out) / "Dental_dataset"

    cs = root / "Childrens dental caries segmentation dataset"
    for split, n, enc in (("Train", 10, 255), ("Test", 4, 1), ("Supplemental content93", 4, 128)):
        for i in range(n):
            img, mask, _ = draw_case(rng)
            stem = f"cate1_{i:03d}" if split != "Supplemental content93" else f"sup_{i:03d}"
            write(cs / split / "images" / f"{stem}.png", img)
            write(cs / split / "mask" / f"{stem}.png", (mask * enc).astype(np.uint8))

    cd = root / "Pediatric dental disease detection dataset"
    for split, n in (("Train", args.n_det_train), ("Test", args.n_det_test)):
        for i in range(n):
            img, mask, teeth = draw_case(rng)
            stem = f"cate1_{i:03d}" if split == "Train" else f"cate2_{i:03d}"
            write(cd / split / "images" / f"{stem}.png", img)
            lm = labelme(teeth, rng, img.shape[1], img.shape[0])
            p = cd / split / "label" / f"{stem}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(lm, ensure_ascii=False))

    ad = root / "Adult tooth segmentation dataset"
    for coll, sub in (("Dataset and code", ("train", "test")), ("Archive", None)):
        dirs = [ad / coll / s for s in sub] if sub else [ad / coll]
        for d in dirs:
            for i in range(4):
                img, mask, _ = draw_case(rng)
                write(d / "images" / f"a{i:03d}.jpg", img)
                mname = "masks" if coll == "Dataset and code" else "mask"
                write(d / mname / f"a{i:03d}.bmp", (mask * 255).astype(np.uint8))
    print("Synthetic dataset written to", root)


if __name__ == "__main__":
    main()
