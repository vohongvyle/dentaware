"""Sanity checks for the detection evaluator.  Run:  python -m pytest -q tests"""

import math

from dentaware.det.evaluate import average_precision, detection_outcomes, evaluate_records, paired_bootstrap_map


def rec(gt, pred, i="a"):
    return {"image_id": i, "width": 100, "height": 100, "gt": gt, "pred": pred}


def test_perfect_prediction_gives_ap_1():
    r = [rec([[10, 10, 30, 30, 0]], [[10, 10, 30, 30, 0.9, 0]])]
    assert average_precision(r, 0) == 1.0


def test_wrong_class_gives_ap_0():
    r = [rec([[10, 10, 30, 30, 0]], [[10, 10, 30, 30, 0.9, 1]])]
    assert average_precision(r, 0) == 0.0


def test_false_positive_ranked_first_halves_precision():
    r = [rec([[10, 10, 30, 30, 0]], [[60, 60, 80, 80, 0.9, 0], [10, 10, 30, 30, 0.8, 0]])]
    assert abs(average_precision(r, 0) - 0.5) < 1e-9


def test_class_without_gt_is_nan_and_excluded_from_map():
    r = [rec([[10, 10, 30, 30, 0]], [[10, 10, 30, 30, 0.9, 0]])]
    out = evaluate_records(r)
    assert math.isnan(out["per_class"]["pulpitis"])
    assert out["mAP"] == 1.0


def test_duplicate_detection_counts_as_false_positive():
    r = [rec([[10, 10, 30, 30, 0]], [[10, 10, 30, 30, 0.9, 0], [11, 11, 31, 31, 0.8, 0]])]
    outs = detection_outcomes(r)
    assert sorted(o["correct"] for o in outs) == [0, 1]


def test_paired_bootstrap_identical_models_zero_diff():
    rs = [rec([[10, 10, 30, 30, 0]], [[10, 10, 30, 30, 0.9, 0]], i=str(k)) for k in range(5)]
    d = paired_bootstrap_map(rs, rs, n=50)
    assert d["diff"] == 0.0
