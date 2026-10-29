"""Calibrated SystemOne loss combining Label-Smoothed CE, Brier, and Ordinal EMD."""

from typing import List

import torch
import torch.nn.functional as F


def compute_calibrated_systemone_loss(
    logits_list: List[torch.Tensor],
    targets_list: List[torch.Tensor],
    question_types: List[str],
    label_smoothing: float = 0.03,
    brier_weight: float = 0.5,
    emd_weight: float = 0.3,
) -> torch.Tensor:
    """
    Combines:
      1. Label-Smoothed Cross-Entropy (prevents logit explosion & overconfidence)
      2. Multi-class Brier Loss (enforces calibrated empirical probabilities)
      3. Earth Mover's Distance / CDF Loss on `score` questions (ordinal partial credit)

    Args:
        logits_list: List of 1D logit tensors per question [num_opts_q].
        targets_list: List of 1D target distribution/one-hot tensors per question [num_opts_q].
        question_types: List of question type strings ("noul" | "choice" | "score").
        label_smoothing: Factor for label smoothing (default: 0.03).
        brier_weight: Weight multiplier for Brier score loss (default: 0.5).
        emd_weight: Weight multiplier for ordinal EMD loss (default: 0.3).

    Returns:
        Scalar torch.Tensor total calibrated loss.
    """
    if not logits_list:
        return torch.tensor(0.0, requires_grad=True)

    device = logits_list[0].device
    total_loss = torch.zeros((), device=device)
    num_q = 0

    for logits, target, q_type in zip(logits_list, targets_list, question_types):
        if logits.numel() == 0 or target.numel() == 0:
            continue

        logits_fp32 = logits.float()
        target_fp32 = target.to(device=logits.device, dtype=torch.float32)
        K = logits_fp32.size(-1)

        # 1. Soft / Label-Smoothed Cross Entropy
        smoothed_target = target_fp32 * (1.0 - label_smoothing) + (label_smoothing / max(K, 1))
        log_probs = F.log_softmax(logits_fp32, dim=-1)
        ce_loss = -(smoothed_target * log_probs).sum()

        # 2. Multi-Class Brier Score Loss: sum((p_i - y_i)^2)
        probs = log_probs.exp()
        brier_loss = F.mse_loss(probs, target_fp32, reduction="mean") * K

        q_loss = ce_loss + brier_weight * brier_loss

        # 3. Ordinal CDF Loss for `score` questions (penalizes distance on ordered scales)
        if q_type == "score" and K > 2:
            cdf_pred = torch.cumsum(probs, dim=-1)
            cdf_true = torch.cumsum(target_fp32, dim=-1)
            emd_loss = F.mse_loss(cdf_pred, cdf_true, reduction="sum")
            q_loss = q_loss + emd_weight * emd_loss

        total_loss = total_loss + q_loss
        num_q += 1

    return total_loss / max(num_q, 1)
