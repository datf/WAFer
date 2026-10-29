"""Evaluation metrics, calibration gates, and benchmarking."""

from wafer.eval.calibration import (
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
    compute_hard_negative_fpr,
    benchmark_single_pass_latency,
)

__all__ = [
    "compute_expected_calibration_error",
    "compute_multiclass_brier_score",
    "compute_hard_negative_fpr",
    "benchmark_single_pass_latency",
]
