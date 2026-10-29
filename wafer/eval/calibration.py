"""Automated Calibration Gates and Latency Benchmarking Suite."""

import time
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from wafer.model.encoder import EncodedRecord


def compute_expected_calibration_error(
    probs: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE):
    Bins predictions by confidence, computes accuracy within each bin, and takes
    the weighted average absolute difference between accuracy and confidence.
    Target threshold: < 0.025.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Confidences and predictions
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)

    # Convert one-hot to class index if necessary
    if labels.ndim > 1:
        ground_truth = np.argmax(labels, axis=-1)
    else:
        ground_truth = labels

    accuracies = (predictions == ground_truth).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_multiclass_brier_score(
    probs: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Computes Multi-Class Brier Score: mean sum((p_i - y_i)^2).
    Target threshold: < 0.08.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Convert categorical 1D labels to one-hot if needed
    if targets.ndim == 1:
        num_classes = probs.shape[-1]
        one_hot = np.zeros((len(targets), num_classes), dtype=float)
        for idx, val in enumerate(targets):
            one_hot[idx, int(val)] = 1.0
        targets = one_hot

    # Mean squared error across classes and samples
    brier = np.mean(np.sum((probs - targets) ** 2, axis=-1))
    return float(brier)


def compute_hard_negative_fpr(
    hard_neg_block_probs: List[float],
    threshold: float = 0.90,
) -> float:
    """
    Computes false-positive rate on hard-negative developer payloads:
    Proportion of benign records predicted as 'block' with confidence >= threshold.
    Target threshold: < 0.05% (0.0005).
    """
    if not hard_neg_block_probs:
        return 0.0
    arr = np.array(hard_neg_block_probs)
    false_blocks = np.sum(arr >= threshold)
    return float(false_blocks / len(arr))


def benchmark_single_pass_latency(
    model: torch.nn.Module,
    sample_rec: EncodedRecord,
    iterations: int = 50,
    warmup: int = 5,
) -> Dict[str, float]:
    """
    Benchmarks single-pass prefill latency (in milliseconds).
    Measures p50, p90, p95, and p99 percentiles.
    """
    model.eval()
    latencies: List[float] = []

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            model.forward_record(sample_rec)

        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            model.forward_record(sample_rec)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

    arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }
