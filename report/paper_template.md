# Tooth-Aware, Reliability-Calibrated Disease Detection in Pediatric Panoramic Radiographs

Vo Hong Vy Le — California State University, Long Beach

> Template for a 6–8 page report or arXiv preprint. Every `[[...]]` is filled from a results file named
> next to it. Do not fill a number you have not run. Report confidence intervals with every estimate.

## Abstract (≈200 words, write last)

Pediatric panoramic radiographs are read visually, and published work on the public Children's Dental
Panoramic Radiographs dataset reports tooth segmentation but no disease-detection baseline. We (1) report
the first detection baseline for six pediatric conditions on this dataset under cross-validation,
(2) test whether tooth awareness improves detection, (3) compare strategies for using adult radiographs
in pediatric tooth segmentation, and (4) measure whether detector confidences are calibrated enough to
refer uncertain cases to a clinician. [[one sentence per result, with CIs]]

## 1. Introduction

* Clinical need: pediatric radiographs, mixed dentition, inter-reader variability.
* Gap: Zhang et al. (2023) released the dataset with segmentation baselines only and found that adding
  adult data did not help pediatric segmentation.
* Contributions: the four research questions (RQ1–RQ4 in the README).

## 2. Related work

Tooth segmentation on panoramic radiographs; caries and lesion detection; small-object detection;
calibration and selective prediction in medical imaging. (Cite only papers you have read.)

## 3. Data

* Subsets and counts → `audit/summary.json`: [[pediatric seg train/test/supplement]],
  [[detection train/test]], [[adult images used]].
* Lesions per class → `audit/class_counts.csv` (table). Lesion sizes → `audit/box_sizes.png`.
* Data problems found and how they were handled (mask encodings, unknown labels, clipped boxes) →
  `audit/summary.json`.

## 4. Methods

4.1 Tooth segmentation: U-Net (ResNet-34 encoder), 512×1024 input, BCE + Dice, validation-only model
selection; three strategies for adult data.

4.2 Detection: YOLO ([[model]]), [[image size]], colour augmentation disabled; 5-fold stratified CV on the
70 training images; final model evaluated once on the 30 test images.

4.3 Tooth awareness: input fusion (tooth probability as third channel); post-hoc filter (threshold chosen
on out-of-fold predictions → `det/filter_threshold.json`); oracle filter with true masks.

4.4 Lesion-to-tooth assignment: watershed tooth instances, jaw split, positional numbering.

4.5 Reliability: ECE (10 bins), cross-fitted isotonic recalibration, referral rule.

4.6 Statistics: image-level bootstrap (1,000 resamples) for 95% CIs; paired bootstrap for differences.

## 5. Results

**Table 1 — Segmentation (RQ3)** → `seg/seg_results.csv`, differences → `seg/seg_summary.json`

| Strategy | Test Dice [95% CI] | Test IoU |
|---|---|---|
| Pediatric only | [[ ]] | [[ ]] |
| Mixed adult + pediatric | [[ ]] | [[ ]] |
| Adult pretrain → pediatric fine-tune | [[ ]] | [[ ]] |

**Table 2 — Detection (RQ1, RQ2)** → `det/det_results.csv`, comparisons → `det/det_summary.json`

| Variant | CV mAP@0.5 [CI] | Test mAP@0.5 [CI] | Small-lesion mAP | Caries AP |
|---|---|---|---|---|
| Baseline | | | | |
| Fused (tooth channel) | | | | |
| Filtered (post-hoc) | | | | |
| Oracle filter (subset) | | | | |

**Figure 1** `figures/examples_*.png` · **Figure 2** `figures/detection_map.png`

**Reliability (RQ4)** → `reliability/reliability_summary.json`: ECE raw [[ ]] → calibrated [[ ]];
`reliability/reliability_test.png`; referral trade-off `reliability/triage.png`.

**Efficiency** → `latency.json`: [[ms per radiograph, GPU, parameters]].

## 6. Discussion

What the CIs allow you to claim and what they do not. Which classes are too rare to evaluate. Whether
tooth awareness helped small lesions. What a negative result means.

## 7. Limitations

100 detection images; one image per child assumed; positional (not FDI) tooth numbering; single dataset
from one source; not a diagnostic device.

## 8. Future work

Expert gaze as supervision: record where dentists look when reading pediatric radiographs (eye tracking)
and compare it with model attention. Requires IRB approval and clinician participants.

## References

Zhang *et al.* (2023), *Scientific Data* — dataset. [[add the papers you cite]]
