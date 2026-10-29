"""Unit tests for runtime schema permutation and anti-memorization."""

import pytest

from wafer.data.permutation import permute_waf_record, PermutedWAFDataset
from wafer.data.synthetic import generate_synthetic_waf_samples
from wafer.model.backbone import MockTokenizer


@pytest.fixture
def synthetic_records():
    return generate_synthetic_waf_samples()


def test_permute_waf_record_modifications(synthetic_records):
    rec = synthetic_records[1]
    orig_options = list(rec["questions"]["action"]["criteria"].keys())

    # Run permutation multiple times; option order should vary
    shuffled_at_least_once = False
    for _ in range(10):
        permuted = permute_waf_record(rec, drop_questions_prob=0.0, paraphrase_prob=0.0)
        new_options = list(permuted["questions"]["action"]["criteria"].keys())
        if new_options != orig_options:
            shuffled_at_least_once = True
            break

    assert shuffled_at_least_once


def test_permuted_dataset_iteration(synthetic_records):
    tokenizer = MockTokenizer()
    dataset = PermutedWAFDataset(synthetic_records, tokenizer, permute=True)

    assert len(dataset) == len(synthetic_records)
    for i in range(len(dataset)):
        encoded = dataset[i]
        assert encoded.input_ids.size(0) > 0
        assert len(encoded.option_spans) > 0
