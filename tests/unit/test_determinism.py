"""Unit test verifying strict determinism and zero output variance across runs."""

import pytest
import torch

from wafer.data.synthetic import generate_synthetic_waf_samples
from wafer.eval.inspect_logs import evaluate_sanitized_request, get_trained_surrogate
from wafer.model.backbone import MockTokenizer, SurrogateDecisionModel
from wafer.model.encoder import encode_record


def test_mock_tokenizer_salt_invariance():
    tok1 = MockTokenizer()
    tok2 = MockTokenizer()

    words = "GET /api/v1/auth/login User-Agent: curl/8.5.0 Host: ha.example.com"
    ids1 = tok1.encode(words)
    ids2 = tok2.encode(words)

    assert ids1 == ids2
    assert len(ids1) > 0


def test_model_inference_determinism():
    torch.manual_seed(42)
    tok = MockTokenizer()
    model = SurrogateDecisionModel(hidden_size=64, d_head=32, n_heads=4, vocab_size=1000)
    model.eval()

    rec = generate_synthetic_waf_samples()[0]
    enc = encode_record(tok, rec)

    with torch.no_grad():
        out1 = model.forward_record(enc)
        out2 = model.forward_record(enc)

    for q1, q2 in zip(out1, out2):
        assert torch.allclose(q1, q2, atol=1e-7)


def test_checkpoint_reproducibility(tmp_path):
    ckpt_path = str(tmp_path / "test_ckpt.pt")

    # Run 1: Train & save
    model1, tok1 = get_trained_surrogate(checkpoint_path=ckpt_path, epochs=5, seed=42, force_retrain=True)

    # Run 2: Load checkpoint
    model2, tok2 = get_trained_surrogate(checkpoint_path=ckpt_path, seed=42, force_retrain=False)

    rec = generate_synthetic_waf_samples()[1]
    eval1 = evaluate_sanitized_request(model1, tok1, rec["state"])
    eval2 = evaluate_sanitized_request(model2, tok2, rec["state"])

    # Probabilities must be identical across both runs
    assert eval1["p_deny"] == eval2["p_deny"]
    assert eval1["p_alert"] == eval2["p_alert"]
    assert eval1["decision"] == eval2["decision"]
    assert eval1["action"] == eval2["action"]
    assert eval1["alert"] == eval2["alert"]
