"""A small web demo: upload a panoramic radiograph, see teeth, findings, and when to refer.

In Colab (after the pipeline has run):

    from dentaware.demo import build_app
    build_app(OUT, DATA).launch(share=True)      # prints a public link

From a terminal:

    python -m dentaware.demo --out runs/main --data-root /path/to/download --share

Research prototype. Not a diagnostic device.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .labels import CLASSES
from .teeth import assign_lesions, split_teeth, tooth_overlap

COLORS_BGR = [(13, 85, 230), (189, 130, 49), (84, 163, 49), (177, 107, 117), (40, 45, 222), (99, 99, 99)]
NICE = {"caries": "Caries", "periapical_lesion": "Periapical lesion", "pulpitis": "Pulpitis",
        "deep_fissure": "Deep fissure", "developmental_anomaly": "Developmental anomaly", "other": "Other"}


class Predictor:
    """Tooth segmentation -> disease detection -> calibration -> tooth assignment -> referral flag.

    seg_fn(gray) -> tooth probability map in [0,1] at the image's size
    det_fn(input_bgr) -> [[x1, y1, x2, y2, score, cls], ...]
    """

    def __init__(self, seg_fn, det_fn, variant: str = "baseline", calibrator=None, threshold: float = 0.4,
                 filter_tau: float = 0.0, referral_band: float = 0.1):
        self.seg_fn, self.det_fn = seg_fn, det_fn
        self.variant, self.calibrator = variant, calibrator
        self.threshold, self.filter_tau, self.referral_band = threshold, filter_tau, referral_band

    def __call__(self, gray: np.ndarray, threshold: float | None = None) -> dict:
        thr = self.threshold if threshold is None else threshold
        prob = self.seg_fn(gray)
        tooth = (prob > 0.5).astype(np.uint8)
        if self.variant == "fused":
            inp = np.dstack([gray, gray, (prob * 255).astype(np.uint8)])
        else:
            inp = np.dstack([gray, gray, gray])
        preds = self.det_fn(inp)
        if self.variant == "filtered":
            preds = [p for p in preds if tooth_overlap(p, tooth) >= self.filter_tau]
        raw = np.array([p[4] for p in preds], dtype=float)
        cal = self.calibrator.predict(raw) if (self.calibrator is not None and raw.size) else raw
        labels, teeth = split_teeth(tooth)
        kept = [(p, float(c)) for p, c in zip(preds, cal) if c >= thr]
        where = assign_lesions([p for p, _ in kept], labels, teeth)
        findings = [{"finding": NICE[CLASSES[int(p[5])]], "tooth": a["tooth"] or "—", "confidence": round(c, 2),
                     "box": [int(v) for v in p[:4]]} for (p, c), a in zip(kept, where)]
        uncertain = int(np.sum(np.abs(cal - thr) < self.referral_band)) if cal.size else 0
        return {"tooth_prob": prob, "tooth_labels": labels, "teeth": teeth, "findings": findings,
                "n_uncertain": uncertain, "refer": uncertain > 0, "threshold": thr}

    @staticmethod
    def render(gray: np.ndarray, result: dict) -> np.ndarray:
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        edges = cv2.Canny((result["tooth_labels"] > 0).astype(np.uint8) * 255, 50, 150)
        img[edges > 0] = (230, 200, 120)
        scale = max(gray.shape) / 1200
        for t in result["teeth"]:
            cv2.putText(img, t["name"], (int(t["cx"]) - 8, int(t["cy"])), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35 * scale, (255, 255, 255), max(1, int(scale)), cv2.LINE_AA)
        for f in result["findings"]:
            k = [NICE[c] for c in CLASSES].index(f["finding"])
            x1, y1, x2, y2 = f["box"]
            col = COLORS_BGR[k % len(COLORS_BGR)]
            cv2.rectangle(img, (x1, y1), (x2, y2), col, max(2, int(2 * scale)))
            cv2.putText(img, f"{f['finding']} {f['confidence']:.2f}", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * scale, col, max(1, int(1.5 * scale)), cv2.LINE_AA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_predictor(out_dir: str | Path, device: str | None = None) -> Predictor:
    """Build the predictor from a finished pipeline run."""
    import pandas as pd
    from sklearn.isotonic import IsotonicRegression

    from .config import Config
    from .seg.experiments import load_tooth_model
    from .seg.model import predict_prob

    out = Path(out_dir)
    dev = device or Config().resolve_device()
    seg_model, size = load_tooth_model(out / "seg" / "tooth_unet.pt", dev)

    rel_path = out / "reliability" / "reliability_summary.json"
    rel = json.loads(rel_path.read_text()) if rel_path.exists() else {}
    variant = rel.get("variant", "baseline")
    threshold = float(rel.get("operating_threshold_from_oof", 0.4))
    weights_variant = "baseline" if variant == "filtered" else variant
    weights = out / "det" / weights_variant / "final" / "train" / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"No detector weights at {weights}. Run the 'det' stage first.")

    calibrator = None
    det_csv = out / "reliability" / "detections_oof.csv"
    if det_csv.exists():
        df = pd.read_csv(det_csv)
        if df["correct"].nunique() > 1:
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(df["score"], df["correct"])

    tau = 0.0
    ft = out / "det" / "filter_threshold.json"
    if variant == "filtered" and ft.exists():
        tau = float(json.loads(ft.read_text())["chosen_threshold"])

    cfg = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else {}
    det_size = int(cfg.get("det_img_size", 1024))

    from ultralytics import YOLO
    yolo = YOLO(str(weights))
    ydev = 0 if dev == "cuda" else dev

    def seg_fn(gray):
        return predict_prob(seg_model, gray, size=size, device=dev)

    def det_fn(bgr):
        r = yolo.predict(source=bgr, imgsz=det_size, conf=0.01, iou=0.6, max_det=100, device=ydev, verbose=False)[0]
        b = r.boxes
        if not len(b):
            return []
        xyxy, conf, cls = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()
        return [[*map(float, xyxy[i]), float(conf[i]), int(cls[i])] for i in range(len(conf))]

    return Predictor(seg_fn, det_fn, variant=variant, calibrator=calibrator, threshold=threshold, filter_tau=tau)


def summary_markdown(result: dict, variant: str) -> str:
    n = len(result["findings"])
    head = f"**{n} finding{'s' if n != 1 else ''}** at confidence ≥ {result['threshold']:.2f} · {len(result['teeth'])} teeth outlined"
    if result["refer"]:
        flag = (f"\n\n⚠️ **Refer to a dentist:** {result['n_uncertain']} detection(s) are close to the decision "
                f"threshold, so the model is not confident about this radiograph.")
    else:
        flag = "\n\n✅ No borderline detections. The model is confident about this radiograph (it can still be wrong)."
    return head + flag + f"\n\n<small>Model: {variant}. Tooth names are jaw + position from the image's left "\
                         f"(U = upper, L = lower), not FDI numbers. Research prototype, not for diagnosis.</small>"


def build_app(out_dir: str | Path, data_root: str | None = None, predictor: Predictor | None = None):
    import gradio as gr

    pred = predictor or load_predictor(out_dir)
    examples = []
    if data_root:
        try:
            from .paths import build_index
            examples = [[str(s.image)] for s in build_index(data_root).child_det if s.split == "test"][:6]
        except Exception:
            examples = []

    def run(image, threshold):
        if image is None:
            return None, "Upload a panoramic radiograph.", []
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        res = pred(gray, threshold)
        table = [[f["finding"], f["tooth"], f["confidence"]] for f in res["findings"]]
        return Predictor.render(gray, res), summary_markdown(res, pred.variant), table

    with gr.Blocks(title="DentAware demo") as app:
        gr.Markdown("# DentAware\nTooth-aware disease detection in pediatric panoramic radiographs. "
                    "Upload an X-ray or pick an example from the test set.")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label="Panoramic radiograph", type="numpy", image_mode="L")
                thr = gr.Slider(0.05, 0.95, value=round(pred.threshold, 2), step=0.05,
                                label="Confidence threshold (default chosen on cross-validation)")
                btn = gr.Button("Analyse", variant="primary")
                if examples:
                    gr.Examples(examples=examples, inputs=inp)
            with gr.Column(scale=2):
                out_img = gr.Image(label="Teeth (outlined) and findings")
                out_md = gr.Markdown()
                out_tab = gr.Dataframe(headers=["Finding", "Tooth", "Confidence"], label="Findings")
        btn.click(run, [inp, thr], [out_img, out_md, out_tab])
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="pipeline output folder, e.g. runs/main")
    ap.add_argument("--data-root", default=None, help="dataset folder, to offer test images as examples")
    ap.add_argument("--share", action="store_true", help="create a public link (useful in Colab)")
    a = ap.parse_args()
    build_app(a.out, a.data_root).launch(share=a.share)


if __name__ == "__main__":
    main()
