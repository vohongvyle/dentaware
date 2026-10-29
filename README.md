# DentAware

**Tooth-aware, reliability-calibrated disease detection in pediatric panoramic radiographs**

Vo Hong Vy Le · California State University, Long Beach

Panoramic X-rays of children are read by eye, and readings vary between clinicians. Public work on the
[Children's Dental Panoramic Radiographs dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC10267170/)
(Zhang et al., *Scientific Data*, 2023) reports tooth-segmentation baselines but **no disease-detection
baseline**, and the authors note that **adding adult images did not improve segmentation on children**.
This project fills both gaps and asks whether a detector can tell when it should defer to a dentist.

## Research questions

| | Question | How it is tested |
|---|---|---|
| **RQ1** | How well can current detectors find six pediatric conditions (caries, periapical lesion, pulpitis, deep fissure, developmental anomaly, other)? | YOLO baseline, 5-fold stratified cross-validation on the 70 training images plus one evaluation on the official 30-image test split; per-class AP@0.5 with bootstrap 95% CIs |
| **RQ2** | Does knowing where the teeth are improve detection, especially of small lesions? | (a) predicted tooth map as an extra input channel; (b) removing detections that lie off the teeth; (c) the same filter with true tooth masks as an upper bound. Paired bootstrap against the baseline, overall and for the smallest third of lesions |
| **RQ3** | How should adult data be used for pediatric tooth segmentation? | pediatric-only vs. mixed adult+pediatric (the authors' setting) vs. adult pretraining then pediatric fine-tuning; Dice on the official pediatric test split, paired bootstrap |
| **RQ4** | Are the detector's confidences trustworthy, and can it refer uncertain cases? | expected calibration error before/after cross-fitted isotonic recalibration; accuracy on automatically handled images vs. share referred to a dentist |

Each detected lesion is also attached to the tooth it sits on (upper/lower jaw and position from the
image's left), because a clinician needs "caries on L4", not just a box.


## What changed from the class project

| Class project (CECS 553) | DentAware |
|---|---|
| Auto-discovery silently trained the U-Net on the **adult** subset | Every subset is located by name and reported; the pediatric set is used for evaluation |
| 192 px input, 300 training images, one random split | Aspect-preserving 512×1024 input, official splits, validation-only model selection |
| Disease boxes were only drawn on the image | Disease detection is trained, cross-validated and evaluated |
| No uncertainty on results | Bootstrap 95% CIs and paired tests everywhere |
| — | Tooth-aware detection, lesion-to-tooth assignment, calibration, triage, latency |

## Run it (Google Colab, free T4 GPU)

Open `notebooks/run_all.ipynb` in Colab and run the cells in order. It installs the requirements,
downloads the dataset with `kagglehub`, and saves results to Google Drive so a disconnected session can
resume. Approximate times on a T4: audit 1 min, segmentation 1–1.5 h, detection 2–3 h (two input
variants × six models), reliability and figures a few minutes.

From a terminal:

```bash
pip install -r requirements.txt
python -m dentaware.pipeline --data-root /path/to/download --out runs/main --stage all
# or one stage at a time: audit | seg | det | reliability | figures | latency
python -m dentaware.pipeline --data-root /path/to/download --out runs/smoke --stage all --quick   # smoke test
```

No GPU? `python tests/smoke_no_gpu.py` checks the data handling, evaluation and analysis code on a
synthetic dataset with stand-ins for the networks.

## Test it

1. **No GPU, 1 minute:** `python tests/smoke_no_gpu.py` checks data handling, evaluation, calibration and
   figures on a synthetic dataset.
2. **Colab, about 10 minutes:** in `notebooks/run_all.ipynb`, run cells 1–4 and the *smoke test* cell
   (`--quick`). It trains tiny models end to end; the numbers mean nothing, it only proves the pipeline runs.
3. **Full study:** run Steps 1–5 in the notebook.

## Web demo

After the pipeline has run, the last notebook cell opens a small web page (Gradio). Upload a panoramic
X-ray or pick a test-set example: the page outlines the teeth, lists each finding with its tooth and
calibrated confidence, and flags radiographs that should be referred to a dentist because some detections
sit close to the decision threshold. In Colab, `share=True` gives a public link while the notebook runs.

```bash
python -m dentaware.demo --out runs/main --data-root /path/to/download --share
```

## Outputs (`runs/main/`)

| File | Contents |
|---|---|
| `audit/summary.json`, `audit/class_counts.csv`, `audit/box_sizes.png` | Subset sizes, mask encodings, label problems, lesions per class, lesion sizes |
| `seg/seg_results.csv`, `seg/seg_summary.json` | Dice/IoU per strategy with 95% CIs and paired differences (RQ3) |
| `det/det_results.csv`, `det/det_summary.json` | mAP@0.5 and per-class AP per variant, cross-validation and test, with CIs and paired comparisons (RQ1, RQ2) |
| `reliability/reliability_summary.json`, `reliability_*.png`, `triage.png` | Calibration and referral analysis (RQ4) |
| `reliability/lesion_teeth_test.csv` | Every confident test detection with its tooth |
| `figures/*.png` | Figures for the report |
| `latency.json` | Milliseconds per radiograph and model size |

## Honest limitations

* The detection subset has **100 radiographs** (70 train / 30 test) and some classes have very few
  boxes, so per-class APs are noisy. Confidence intervals are reported for this reason; read them
  before reading the point estimates.
* Each radiograph is treated as one child (the dataset describes 100 curated patients).
* Tooth positions are counted from the image's left within each jaw. They are not FDI numbers, because
  the dataset has no tooth-identity labels to validate them against.
* This is a research prototype, not a diagnostic device.

## Citation for the data

Zhang *et al.* Children's dental panoramic radiographs dataset for caries segmentation and dental
disease detection. *Scientific Data* (2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10267170/
(copy the full author list and article number from that page before publishing).
