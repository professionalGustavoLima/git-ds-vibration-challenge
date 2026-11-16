from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.features import F_MIN_HZ
from src.interface import CarpetRegion

Interval = Tuple[float, float]

_MATCH_IOU = 0.5  # minimum IoU for a predicted region to count as finding a labelled one


def _to_intervals(regions: Sequence[CarpetRegion]) -> List[Interval]:
    return [(float(r.start_hz), float(r.end_hz)) for r in regions]


def _merge(intervals: Sequence[Interval]) -> List[Interval]:
    """Sorted union of intervals (overlapping or touching intervals become one)."""
    merged: List[List[float]] = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _length(intervals: Sequence[Interval]) -> float:
    return float(sum(b - a for a, b in intervals))


def _intersection_length(x: Sequence[Interval], y: Sequence[Interval]) -> float:
    """Total length of the intersection of two merged (sorted, disjoint) interval lists."""
    total, i, j = 0.0, 0, 0
    while i < len(x) and j < len(y):
        total += max(0.0, min(x[i][1], y[j][1]) - max(x[i][0], y[j][0]))
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return total


def _interval_iou(p: Interval, t: Interval) -> float:
    inter = max(0.0, min(p[1], t[1]) - max(p[0], t[0]))
    union = (p[1] - p[0]) + (t[1] - t[0]) - inter
    return inter / union if union > 0 else 0.0


def _match_regions(pred: Sequence[Interval], true: Sequence[Interval]) -> List[Tuple[int, int]]:
    """One-to-one matching maximising total IoU (Hungarian); only pairs with IoU >= ``_MATCH_IOU`` are kept."""
    if not pred or not true:
        return []
    iou = np.array([[_interval_iou(p, t) for t in true] for p in pred])
    rows, cols = linear_sum_assignment(iou, maximize=True)
    return [(i, j) for i, j in zip(rows, cols) if iou[i, j] >= _MATCH_IOU]


def _constraint_violations(regions: Sequence[CarpetRegion]) -> int:
    """Number of predicted regions breaking requirement b (start above 1000 Hz), d (no overlap) or start < end."""
    intervals = sorted(_to_intervals(regions))
    bad = sum(1 for a, b in intervals if a <= F_MIN_HZ or b <= a)
    bad += sum(1 for (_, b0), (a1, _) in zip(intervals, intervals[1:]) if a1 < b0)
    return bad


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def _f1(precision: float, recall: float) -> float:
    if np.isnan(precision) or np.isnan(recall):
        return float("nan")
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_part1(
    predictions: dict[str, List[CarpetRegion]],
    labels: dict[str, List[CarpetRegion]],
) -> dict:
    """Evaluate carpet detector predictions against ground truth.

    Regions are compared as exact frequency intervals (Hz). The regions of each signal are merged first, which
    also resolves the overlapping labels of two samples (they break requirement d).

    Returned metrics:

    - ``mean_iou`` (main metric): IoU in Hz per signal, averaged over signals. A signal without carpet scores 1
      when nothing is predicted and 0 when anything is.
    - ``mean_iou_carpet_signals``: the same average, only over signals that have carpet.
    - ``clean_signal_accuracy``: fraction of signals without carpet where nothing was predicted.
    - ``hz_precision``, ``hz_recall``, ``hz_f1``: Hz overlap pooled over all signals.
    - ``region_precision``, ``region_recall``, ``region_f1``: one-to-one region matching (Hungarian, IoU >= 0.5).
    - ``mean_boundary_error_hz``: mean absolute start/end error of the matched regions.
    - ``constraint_violations``: predicted regions breaking requirement b or d (should be 0).
    - ``n_signals``, ``n_true_regions``, ``n_pred_regions``: counts (after merging).
    """
    pred_ids, label_ids = set(predictions), set(labels)
    if pred_ids != label_ids:
        raise ValueError(
            f"Mismatch between predictions and labels.\n"
            f"  In predictions but not labels: {pred_ids - label_ids}\n"
            f"  In labels but not predictions: {label_ids - pred_ids}"
        )

    ious, carpet_ious, clean_ok = [], [], []
    tp_hz = pred_hz = true_hz = 0.0
    n_pred = n_true = n_matched = violations = 0
    boundary_errors: List[float] = []

    for sample_id, true_regions in labels.items():
        pred = _merge(_to_intervals(predictions[sample_id]))
        true = _merge(_to_intervals(true_regions))
        violations += _constraint_violations(predictions[sample_id])

        inter = _intersection_length(pred, true)
        p_len, t_len = _length(pred), _length(true)
        union = p_len + t_len - inter
        iou = 1.0 if union == 0 else inter / union
        ious.append(iou)
        if true:
            carpet_ious.append(iou)
        else:
            clean_ok.append(not pred)

        tp_hz, pred_hz, true_hz = tp_hz + inter, pred_hz + p_len, true_hz + t_len

        matches = _match_regions(pred, true)
        n_pred, n_true, n_matched = n_pred + len(pred), n_true + len(true), n_matched + len(matches)
        for i, j in matches:
            boundary_errors += [abs(pred[i][0] - true[j][0]), abs(pred[i][1] - true[j][1])]

    hz_precision, hz_recall = _ratio(tp_hz, pred_hz), _ratio(tp_hz, true_hz)
    region_precision, region_recall = _ratio(n_matched, n_pred), _ratio(n_matched, n_true)

    result = {
        "mean_iou": float(np.mean(ious)),
        "mean_iou_carpet_signals": float(np.mean(carpet_ious)) if carpet_ious else float("nan"),
        "clean_signal_accuracy": float(np.mean(clean_ok)) if clean_ok else float("nan"),
        "hz_precision": hz_precision,
        "hz_recall": hz_recall,
        "hz_f1": _f1(hz_precision, hz_recall),
        "region_precision": region_precision,
        "region_recall": region_recall,
        "region_f1": _f1(region_precision, region_recall),
        "mean_boundary_error_hz": float(np.mean(boundary_errors)) if boundary_errors else float("nan"),
        "constraint_violations": violations,
        "n_signals": len(labels),
        "n_true_regions": n_true,
        "n_pred_regions": n_pred,
    }
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}


def summarize_part2_predictions(predictions: dict[str, bool]) -> dict:
    """Counts of unlabelled predictions (e.g. ``test_data``), with the same names used by ``evaluate_part2``.

    Without labels there is no accuracy: ``n_starved`` and ``n_healthy`` are *predicted* counts.
    """
    n_starved = sum(bool(p) for p in predictions.values())
    return {"n_assets": len(predictions), "n_starved": n_starved, "n_healthy": len(predictions) - n_starved}


def evaluate_part2(
    predictions: dict[str, bool],
    labels: dict[str, bool],
) -> dict:
    """Evaluate lubrication predictions against ground truth.

    Each asset is one sample. Predictions for assets without a label (e.g. the two ``test …`` folders under
    ``assets/``) are ignored and counted in ``n_unlabeled_ignored``; a labelled asset without a prediction raises
    ``ValueError``.

    Returned metrics:

    - ``n_assets``: number of labelled assets.
    - ``n_starved``, ``n_healthy``: labelled counts of each class.
    - ``n_unlabeled_ignored``: predictions without a label.
    """
    missing = set(labels) - set(predictions)
    if missing:
        raise ValueError(
            f"Mismatch between predictions and labels.\n"
            f"  In labels but not predictions: {missing}"
        )
    unlabeled = set(predictions) - set(labels)
    n_starved = sum(bool(t) for t in labels.values())

    return {
        "n_assets": len(labels),
        "n_starved": n_starved,
        "n_healthy": len(labels) - n_starved,
        "n_unlabeled_ignored": len(unlabeled),
    }
