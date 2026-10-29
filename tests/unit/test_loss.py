"""Unit tests for calibrated SystemOne loss."""

import pytest
import torch

from wafer.model.loss import compute_calibrated_systemone_loss


def test_compute_calibrated_loss_computation():
    # 2 questions: one choice (3 options), one score (4 options)
    logits_q1 = torch.tensor([2.0, -1.0, 0.5], requires_grad=True)
    target_q1 = torch.tensor([1.0, 0.0, 0.0])

    logits_q2 = torch.tensor([0.1, 1.5, 0.2, -0.5], requires_grad=True)
    target_q2 = torch.tensor([0.0, 1.0, 0.0, 0.0])

    logits_list = [logits_q1, logits_q2]
    targets_list = [target_q1, target_q2]
    q_types = ["choice", "score"]

    loss = compute_calibrated_systemone_loss(
        logits_list=logits_list,
        targets_list=targets_list,
        question_types=q_types,
        label_smoothing=0.03,
        brier_weight=0.5,
        emd_weight=0.3,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0.0

    # Backpropagation check
    loss.backward()
    assert logits_q1.grad is not None
    assert logits_q2.grad is not None


def test_ordinal_emd_loss_penalty():
    """Score question should penalize predictions further from true label more heavily."""
    target = torch.tensor([1.0, 0.0, 0.0, 0.0])  # True = 0

    # Close prediction (class 1)
    logits_close = torch.tensor([0.0, 5.0, 0.0, 0.0])
    loss_close = compute_calibrated_systemone_loss(
        [logits_close], [target], ["score"], emd_weight=1.0
    )

    # Far prediction (class 3)
    logits_far = torch.tensor([0.0, 0.0, 0.0, 5.0])
    loss_far = compute_calibrated_systemone_loss(
        [logits_far], [target], ["score"], emd_weight=1.0
    )

    # EMD penalty must make far prediction loss strictly greater than close prediction loss
    assert loss_far.item() > loss_close.item()
