"""Supervised Stage 1 Training Loop for Gemma-Clef Joint Schema Head."""

import math
import random
from typing import Any, Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from wafer.model.backbone import GemmaClefDecisionModel
from wafer.model.encoder import EncodedRecord
from wafer.model.loss import compute_calibrated_systemone_loss


def get_optimizer_with_differential_lrs(
    model: torch.nn.Module,
    head_lr: float = 2e-4,
    lora_lr: float = 5e-5,
    weight_decay: float = 0.01,
    use_8bit: bool = False,
) -> torch.optim.Optimizer:
    """
    Configures AdamW with differential learning rates:
    - JointSchemaHead parameters: head_lr (2e-4)
    - LoRA adapter parameters: lora_lr (5e-5)
    If use_8bit and CUDA is available, uses bitsandbytes.optim.PagedAdamW8bit.
    """
    head_params = []
    lora_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "joint_head" in name:
            head_params.append(param)
        elif "lora" in name:
            lora_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay})
    if other_params:
        param_groups.append({"params": other_params, "lr": head_lr, "weight_decay": weight_decay})

    if use_8bit and torch.cuda.is_available():
        try:
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW8bit(param_groups)
        except Exception:
            pass

    return AdamW(param_groups)


def train_one_epoch(
    model: torch.nn.Module,
    dataset: List[EncodedRecord],
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int = 4,
    device: Optional[torch.device] = None,
    indices: Optional[List[int]] = None,
) -> float:
    """
    Trains for one epoch over encoded records with gradient accumulation.
    """
    model.train()
    total_loss = 0.0
    accumulated_loss = 0.0
    optimizer.zero_grad()

    num_records = len(dataset)
    step_indices = indices if indices is not None else list(range(num_records))

    for step, idx in enumerate(step_indices):
        rec = dataset[idx]
        logits = model.forward_record(rec)
        if rec.targets is None:
            continue

        loss = compute_calibrated_systemone_loss(
            logits_list=logits,
            targets_list=rec.targets,
            question_types=rec.question_types,
        )

        loss_scaled = loss / gradient_accumulation_steps
        loss_scaled.backward()

        accumulated_loss += loss.item()

        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == num_records:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()
            total_loss += accumulated_loss
            accumulated_loss = 0.0

    return total_loss / max(num_records, 1)


def train_gemma_clef(
    model: torch.nn.Module,
    train_data: List[EncodedRecord],
    epochs: int = 5,
    head_lr: float = 2e-4,
    lora_lr: float = 5e-5,
    gradient_accumulation_steps: int = 4,
    seed: int = 42,
    shuffle: bool = True,
) -> List[float]:
    """Runs full training recipe over multiple epochs with per-epoch dataset shuffling."""
    optimizer = get_optimizer_with_differential_lrs(model, head_lr=head_lr, lora_lr=lora_lr)
    epoch_losses = []

    rng = random.Random(seed)
    indices = list(range(len(train_data)))

    for epoch in range(epochs):
        if shuffle:
            rng.shuffle(indices)

        avg_loss = train_one_epoch(
            model=model,
            dataset=train_data,
            optimizer=optimizer,
            gradient_accumulation_steps=gradient_accumulation_steps,
            indices=indices,
        )
        epoch_losses.append(avg_loss)
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"  [Epoch {epoch+1:2d}/{epochs}] Loss: {avg_loss:.4f}", flush=True)

    return epoch_losses
