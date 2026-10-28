"""Run the whole study, or one stage at a time.

    python -m dentaware.pipeline --data-root /path/to/data --out runs/main --stage all
    python -m dentaware.pipeline --data-root ... --stage audit
    python -m dentaware.pipeline --data-root ... --stage seg
    python -m dentaware.pipeline --data-root ... --stage det
    python -m dentaware.pipeline --data-root ... --stage reliability
    python -m dentaware.pipeline --data-root ... --stage figures
    python -m dentaware.pipeline --data-root ... --stage latency
    python -m dentaware.pipeline --data-root ... --stage all --quick     # 5-minute smoke test
"""

from __future__ import annotations

import argparse
import json
import time

from .config import Config

STAGES = ("audit", "seg", "det", "reliability", "figures", "latency")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--stage", default="all", choices=("all",) + STAGES)
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    ap.add_argument("--det-model", default=None, help="e.g. yolo11s.pt, yolo11m.pt, rtdetr-l.pt")
    ap.add_argument("--det-epochs", type=int, default=None)
    ap.add_argument("--seg-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    cfg = Config(data_root=args.data_root, out_dir=args.out, quick=args.quick, seed=args.seed)
    if args.det_model:
        cfg.det_model = args.det_model
    cfg.apply_quick()
    if args.det_epochs:
        cfg.det_epochs = args.det_epochs
    if args.seg_epochs:
        cfg.seg_epochs = args.seg_epochs
    cfg.save()

    stages = STAGES if args.stage == "all" else (args.stage,)
    t0 = time.time()
    for st in stages:
        print(f"\n######## {st} ########")
        if st == "audit":
            from .audit import run_audit
            print(json.dumps(run_audit(cfg)["detection"], indent=2, ensure_ascii=False))
        elif st == "seg":
            from .seg.experiments import run_seg_experiments
            run_seg_experiments(cfg)
        elif st == "det":
            from .det.run import run_detection
            run_detection(cfg)
        elif st == "reliability":
            from .reliability import run_reliability
            run_reliability(cfg)
        elif st == "figures":
            from .viz import figure_detection_comparison, figure_examples, figure_segmentation_comparison
            from .io_utils import load_json
            rel = load_json(cfg.out / "reliability" / "reliability_summary.json")
            print(figure_segmentation_comparison(cfg))
            print(figure_detection_comparison(cfg))
            print(figure_examples(cfg, rel["variant"], threshold=rel["operating_threshold_from_oof"]))
        elif st == "latency":
            from .latency import run_latency
            run_latency(cfg)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results in {cfg.out}")


if __name__ == "__main__":
    main()
