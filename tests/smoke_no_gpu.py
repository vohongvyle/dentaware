"""End-to-end smoke test that needs neither a GPU nor PyTorch.

It builds a synthetic dataset, stands in for the U-Net with an Otsu threshold and for YOLO with a
noisy copy of the ground truth, then runs audit -> detection protocol -> reliability -> figures.
The numbers it prints are meaningless; it only proves the plumbing works.

    python tests/smoke_no_gpu.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dentaware.audit import run_audit  # noqa: E402
from dentaware.config import Config  # noqa: E402
from dentaware.det.backend import FakeBackend  # noqa: E402
from dentaware.det.run import run_detection  # noqa: E402
from dentaware.io_utils import read_gray, write_image  # noqa: E402
from dentaware.labels import parse_labelme  # noqa: E402
from dentaware.paths import build_index  # noqa: E402
from dentaware.reliability import run_reliability  # noqa: E402
from dentaware.viz import figure_detection_comparison, figure_examples  # noqa: E402


def main():
    tmp = Path(tempfile.mkdtemp(prefix="dentaware_smoke_"))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic.py"), "--out", str(tmp / "data")], check=True)
    cfg = Config(data_root=str(tmp / "data"), out_dir=str(tmp / "run"), quick=True).apply_quick()
    print(run_audit(cfg)["detection"])

    idx = build_index(cfg.data_root)
    gts = {}
    for s in idx.child_det:
        g = read_gray(s.image)
        _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        write_image(cfg.out / "tooth_masks" / s.split / f"{s.image.stem}.png", m)
        h, w = g.shape
        gts[s.id] = parse_labelme(s.label, w, h)[0]

    run_detection(cfg, backend=FakeBackend(gts))
    rel = run_reliability(cfg)
    print(figure_detection_comparison(cfg))
    print(figure_examples(cfg, rel["variant"], n=2, threshold=rel["operating_threshold_from_oof"]))
    print("\nSMOKE TEST PASSED. Outputs in", cfg.out)
    if "--keep" not in sys.argv:
        shutil.rmtree(tmp / "data")


if __name__ == "__main__":
    main()
