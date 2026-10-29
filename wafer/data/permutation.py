"""Runtime schema permutation and anti-memorization data augmentations."""

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from wafer.model.encoder import EncodedRecord, encode_record


# Standard instruction paraphrases for default WAF questions
PARAPHRASE_BANK: Dict[str, List[str]] = {
    "action": [
        "Determine the gateway enforcement action for this incoming HTTP request.",
        "Should the edge gateway allow or deny this request?",
        "Select the appropriate proxy enforcement policy: allow or deny.",
        "Decide whether to pass the request through or block it with 403 Forbidden.",
        "What action should the gateway take on this incoming transaction?",
    ],
    "alert": [
        "Should a security alert or SOC notification be raised for this request?",
        "Determine if this request requires triggering an alert notification.",
        "Does this HTTP request require raising a security alert?",
        "Assess whether this transaction should be flagged for security alerting.",
        "Should this request notify security operations?",
    ],
}


def permute_waf_record(
    record: Dict[str, Any],
    drop_questions_prob: float = 0.3,
    drop_instructions_prob: float = 0.1,
    paraphrase_prob: float = 0.5,
) -> Dict[str, Any]:
    """
    Applies dynamic anti-memorization augmentations to a WAF record:
    1. Question subset & order permutation.
    2. Option order permutation for choice/noul questions.
    3. Instruction paraphrasing and optional instruction omission.
    """
    rec = copy.deepcopy(record)
    questions = rec.get("questions", {})
    targets = rec.get("targets", {})

    q_keys = list(questions.keys())
    random.shuffle(q_keys)

    # Optionally drop 1-2 questions if more than 2 are present
    if len(q_keys) > 2 and random.random() < drop_questions_prob:
        num_to_drop = random.randint(1, min(2, len(q_keys) - 2))
        q_keys = q_keys[num_to_drop:]

    permuted_questions: Dict[str, Any] = {}
    permuted_targets: Dict[str, Any] = {}

    for q_id in q_keys:
        q_spec = questions[q_id]
        q_type = q_spec.get("type", "choice")
        new_q_spec = copy.deepcopy(q_spec)

        # 1. Instruction Augmentation
        if random.random() < drop_instructions_prob:
            # Fall back to question key alone
            new_q_spec.pop("instructions", None)
        elif random.random() < paraphrase_prob and q_id in PARAPHRASE_BANK:
            new_q_spec["instructions"] = random.choice(PARAPHRASE_BANK[q_id])

        # 2. Option Permutation (choice & noul only; keep score ordered)
        criteria = q_spec.get("criteria")
        if q_type == "choice" and isinstance(criteria, dict):
            items = list(criteria.items())
            random.shuffle(items)
            new_q_spec["criteria"] = dict(items)
            # If target was a soft probability list, remap according to permutation
            if q_id in targets:
                t_val = targets[q_id]
                if isinstance(t_val, list):
                    orig_keys = list(criteria.keys())
                    new_keys = [k for k, _ in items]
                    key_to_prob = dict(zip(orig_keys, t_val))
                    permuted_targets[q_id] = [key_to_prob.get(k, 0.0) for k in new_keys]
                else:
                    permuted_targets[q_id] = t_val
        elif q_type == "noul" and isinstance(criteria, dict):
            items = list(criteria.items())
            random.shuffle(items)
            new_q_spec["criteria"] = dict(items)
            if q_id in targets:
                t_val = targets[q_id]
                if isinstance(t_val, list):
                    orig_keys = list(criteria.keys())
                    new_keys = [k for k, _ in items]
                    key_to_prob = dict(zip(orig_keys, t_val))
                    permuted_targets[q_id] = [key_to_prob.get(k, 0.0) for k in new_keys]
                else:
                    permuted_targets[q_id] = t_val
        else:
            # For score or unmodified types, retain target directly
            if q_id in targets:
                permuted_targets[q_id] = targets[q_id]

        permuted_questions[q_id] = new_q_spec

    rec["questions"] = permuted_questions
    rec["targets"] = permuted_targets
    return rec


class PermutedWAFDataset(Dataset):
    """PyTorch Dataset that dynamically permutes records and packs them via encode_record."""
    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: Any,
        permute: bool = True,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.permute = permute

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> EncodedRecord:
        record = self.records[idx]
        if self.permute:
            record = permute_waf_record(record)
        return encode_record(self.tokenizer, record, targets=record.get("targets"))
