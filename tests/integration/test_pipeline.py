"""Integration tests for end-to-end forward pass, training loop, and calibration metrics."""

import pytest
import torch

from wafer.data.synthetic import generate_synthetic_waf_samples
from wafer.eval.calibration import (
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
    compute_hard_negative_fpr,
    benchmark_single_pass_latency,
)
from wafer.model.backbone import MockTokenizer, SurrogateDecisionModel
from wafer.model.encoder import encode_record
from wafer.training.train import train_one_epoch, get_optimizer_with_differential_lrs


@pytest.fixture
def dataset_encoded():
    records = generate_synthetic_waf_samples()
    tokenizer = MockTokenizer()
    return [encode_record(tokenizer, r, targets=r.get("targets")) for r in records]


@pytest.fixture
def surrogate_model():
    return SurrogateDecisionModel(hidden_size=64, d_head=32, n_heads=4, vocab_size=1000)


def test_surrogate_model_forward(surrogate_model, dataset_encoded):
    rec = dataset_encoded[0]
    logits = surrogate_model.forward_record(rec)

    assert isinstance(logits, list)
    assert len(logits) == len(rec.option_spans)
    for q_logits, q_spans in zip(logits, rec.option_spans):
        assert q_logits.shape == (len(q_spans),)


def test_training_one_epoch(surrogate_model, dataset_encoded):
    optimizer = get_optimizer_with_differential_lrs(surrogate_model, head_lr=1e-3, lora_lr=1e-3)
    initial_loss = train_one_epoch(
        surrogate_model, dataset_encoded, optimizer, gradient_accumulation_steps=2
    )

    assert isinstance(initial_loss, float)
    assert initial_loss > 0.0


def test_calibration_metrics():
    # Synthetic well-calibrated probabilities
    probs = torch.tensor([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.1, 0.9],
        [0.2, 0.8],
    ])
    labels = torch.tensor([0, 0, 1, 1])

    ece = compute_expected_calibration_error(probs, labels, n_bins=5)
    assert 0.0 <= ece <= 1.0

    brier = compute_multiclass_brier_score(probs, labels)
    assert 0.0 <= brier <= 2.0

    # Hard negative FPR: test with zero false blocks vs one false block
    fpr_zero = compute_hard_negative_fpr([0.1, 0.05, 0.2], threshold=0.90)
    assert fpr_zero == 0.0

    fpr_one = compute_hard_negative_fpr([0.95, 0.1, 0.2], threshold=0.90)
    assert fpr_one == pytest.approx(1.0 / 3.0)


def test_latency_benchmarking(surrogate_model, dataset_encoded):
    rec = dataset_encoded[0]
    stats = benchmark_single_pass_latency(surrogate_model, rec, iterations=10, warmup=2)

    assert "mean_ms" in stats
    assert "p50_ms" in stats
    assert "p95_ms" in stats
    assert stats["p95_ms"] > 0.0
