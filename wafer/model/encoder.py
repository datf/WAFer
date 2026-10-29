"""Record encoding and token span tracking for Gemma-Clef."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


@dataclass
class EncodedRecord:
    """Represents a packed sequence with recorded token index spans."""
    input_ids: torch.Tensor                     # 1D tensor [seq_len]
    state_span: Tuple[int, int]                 # (start_tok_idx, end_tok_idx)
    option_spans: List[List[Tuple[int, int]]]   # Per question -> list of (start, end) per option
    question_types: List[str]                   # List of "noul" | "choice" | "score"
    targets: Optional[List[torch.Tensor]] = None # List of 1D target tensors per question


def encode_record(
    tokenizer: Any,
    record: Dict[str, Any],
    targets: Optional[Dict[str, Any]] = None,
) -> EncodedRecord:
    """
    Packs `state` and `questions` into a single flat token sequence while recording
    the exact token index spans [start, end) for the state and for every option
    of every question.

    Args:
        tokenizer: HuggingFace or surrogate tokenizer implementing encode().
        record: Dictionary containing "state" and "questions".
        targets: Optional ground-truth targets (overrides record.get("targets")).

    Returns:
        EncodedRecord containing input_ids, state_span, option_spans, question_types, targets.
    """
    # If targets not passed explicitly, inspect record
    if targets is None and "targets" in record:
        targets = record["targets"]

    ids: List[int] = []
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        ids.append(bos_id)

    def append_text(text: str) -> Tuple[int, int]:
        toks = tokenizer.encode(text, add_special_tokens=False)
        start = len(ids)
        ids.extend(toks)
        end = len(ids)
        return start, end

    # 1. Encode State (simplified inputs: method, uri, protocol, user_agent)
    append_text("<|state_start|>\n")
    state_data = record["state"]
    if hasattr(state_data, "to_model_state"):
        state_dict = state_data.to_model_state()
    elif isinstance(state_data, dict):
        state_dict = {
            "method": state_data.get("method", "GET"),
            "uri": state_data.get("uri", "/"),
            "protocol": state_data.get("protocol") or "HTTP/1.1",
            "user_agent": state_data.get("user_agent") or (state_data.get("headers", {}).get("User-Agent", "") if isinstance(state_data.get("headers"), dict) else ""),
        }
    else:
        state_dict = state_data

    if isinstance(state_dict, str):
        state_str = state_dict
    else:
        state_str = json.dumps(state_dict, ensure_ascii=False)

    state_span = append_text(state_str)
    append_text("\n<|state_end|>\n<|schema_start|>\n")

    # 2. Encode Questions & Track Option Spans
    option_spans: List[List[Tuple[int, int]]] = []
    question_types: List[str] = []
    target_tensors: Optional[List[torch.Tensor]] = [] if targets is not None else None

    for q_id, q_spec in record["questions"].items():
        q_type = q_spec.get("type", "choice")
        question_types.append(q_type)
        instr = q_spec.get("instructions", q_id)
        append_text(f"\n[Q: {q_id} | type={q_type}] {instr}\nOptions:\n")

        # Normalize options into a list of (opt_key, opt_description)
        criteria = q_spec.get("criteria")
        if q_type == "noul":
            if isinstance(criteria, dict):
                opts = [
                    ("true", criteria.get("true", "Yes / True")),
                    ("false", criteria.get("false", "No / False")),
                ]
            else:
                opts = [("true", "Yes / True"), ("false", "No / False")]
        elif q_type == "choice":
            if isinstance(criteria, dict):
                opts = list(criteria.items())
            elif isinstance(criteria, list):
                opts = [(str(i), c) for i, c in enumerate(criteria)]
            else:
                raise ValueError(f"Invalid criteria format for choice question: {criteria}")
        elif q_type == "score":
            if isinstance(criteria, list):
                opts = [(str(idx), desc) for idx, desc in enumerate(criteria)]
            elif isinstance(criteria, dict):
                opts = [(str(k), desc) for k, desc in criteria.items()]
            else:
                raise ValueError(f"Invalid criteria format for score question: {criteria}")
        else:
            raise ValueError(f"Unknown question type: {q_type}")

        q_opt_spans: List[Tuple[int, int]] = []
        for opt_key, opt_desc in opts:
            append_text("  - ")
            span = append_text(f"{opt_key}: {opt_desc}")
            append_text("\n")
            # Ensure non-empty span
            if span[0] == span[1]:
                span = (span[0], span[0] + 1)
            q_opt_spans.append(span)

        option_spans.append(q_opt_spans)

        # Map targets if provided
        if target_tensors is not None and targets is not None and q_id in targets:
            t_val = targets[q_id]
            num_opts = len(opts)
            if isinstance(t_val, list):
                # Soft probability distribution
                target_tensors.append(torch.tensor(t_val, dtype=torch.float32))
            else:
                # Hard label (match option key)
                opt_keys = [str(k).lower() for k, _ in opts]
                query_key = str(t_val).lower()
                if query_key in opt_keys:
                    idx = opt_keys.index(query_key)
                else:
                    # Try int conversion if key is integer
                    try:
                        idx = int(t_val)
                    except ValueError:
                        idx = 0
                one_hot = torch.zeros(num_opts, dtype=torch.float32)
                one_hot[idx] = 1.0
                target_tensors.append(one_hot)

    append_text("<|schema_end|>")

    return EncodedRecord(
        input_ids=torch.tensor(ids, dtype=torch.long),
        state_span=state_span,
        option_spans=option_spans,
        question_types=question_types,
        targets=target_tensors,
    )
