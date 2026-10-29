"""Unit tests for record encoding and token span tracking."""

import json
import pytest
import torch

from wafer.model.backbone import MockTokenizer
from wafer.model.encoder import EncodedRecord, encode_record


@pytest.fixture
def tokenizer():
    return MockTokenizer()


@pytest.fixture
def sample_record():
    return {
        "state": {
            "method": "POST",
            "uri": "/api/v1/auth/login",
            "headers": {"Content-Type": "application/json"},
            "body": "{\"user\": \"admin' OR 1=1--\"}",
        },
        "questions": {
            "alert": {
                "type": "noul",
                "instructions": "Should this request trigger an alert?",
                "criteria": {"true": "Yes", "false": "No"},
            },
            "primary_attack": {
                "type": "choice",
                "instructions": "Identify the primary vector.",
                "criteria": {"benign": "Normal traffic", "sqli": "SQL Injection"},
            },
            "severity": {
                "type": "score",
                "instructions": "Rate severity from 0 to 2.",
                "criteria": ["0: Low", "1: Medium", "2: High"],
            },
        },
        "targets": {
            "alert": "true",
            "primary_attack": "sqli",
            "severity": 2,
        },
    }


def test_encode_record_structure(tokenizer, sample_record):
    rec = encode_record(tokenizer, sample_record)

    assert isinstance(rec, EncodedRecord)
    assert isinstance(rec.input_ids, torch.Tensor)
    assert rec.input_ids.ndim == 1
    assert rec.input_ids.size(0) > 0

    # Verify state span
    s_start, s_end = rec.state_span
    assert 0 <= s_start < s_end <= rec.input_ids.size(0)

    # Verify question count
    assert len(rec.option_spans) == 3
    assert rec.question_types == ["noul", "choice", "score"]

    # Verify option spans
    # Question 1: noul -> 2 options
    assert len(rec.option_spans[0]) == 2
    # Question 2: choice -> 2 options
    assert len(rec.option_spans[1]) == 2
    # Question 3: score -> 3 options
    assert len(rec.option_spans[2]) == 3

    for q_spans in rec.option_spans:
        for (o_start, o_end) in q_spans:
            assert 0 <= o_start < o_end <= rec.input_ids.size(0)


def test_encode_record_targets_one_hot(tokenizer, sample_record):
    rec = encode_record(tokenizer, sample_record)

    assert rec.targets is not None
    assert len(rec.targets) == 3

    # alert: "true" -> index 0 in [true, false]
    assert torch.equal(rec.targets[0], torch.tensor([1.0, 0.0]))

    # primary_attack: "sqli" -> index 1 in [benign, sqli]
    assert torch.equal(rec.targets[1], torch.tensor([0.0, 1.0]))

    # severity: 2 -> index 2 in [0, 1, 2]
    assert torch.equal(rec.targets[2], torch.tensor([0.0, 0.0, 1.0]))


def test_encode_record_soft_targets(tokenizer, sample_record):
    sample_record["targets"] = {
        "alert": [0.8, 0.2],
        "primary_attack": [0.1, 0.9],
        "severity": [0.05, 0.15, 0.80],
    }
    rec = encode_record(tokenizer, sample_record)

    assert rec.targets is not None
    assert torch.allclose(rec.targets[0], torch.tensor([0.8, 0.2]))
    assert torch.allclose(rec.targets[1], torch.tensor([0.1, 0.9]))
    assert torch.allclose(rec.targets[2], torch.tensor([0.05, 0.15, 0.80]))
