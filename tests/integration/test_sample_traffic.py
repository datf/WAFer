"""Integration test verifying end-to-end evaluation on real Caddy sample_traffic.log."""

import os
import pytest
from wafer.eval.inspect_logs import get_trained_surrogate, inspect_log_stream


@pytest.fixture(scope="module")
def sample_traffic_lines():
    log_path = "data/sample_traffic.log"
    assert os.path.exists(log_path), f"Log file not found at {log_path}"
    with open(log_path, "r", encoding="utf-8") as f:
        return f.readlines()


@pytest.fixture(scope="module")
def trained_surrogate():
    model, tokenizer = get_trained_surrogate(
        checkpoint_path="checkpoints/calibrated_surrogate.pt",
        epochs=20,
        force_retrain=False,
    )
    return model, tokenizer


def test_sample_traffic_evaluates_without_errors(sample_traffic_lines, trained_surrogate):
    model, tokenizer = trained_surrogate
    evals = inspect_log_stream(sample_traffic_lines, model=model, tokenizer=tokenizer)

    assert len(evals) == len(sample_traffic_lines)
    for res in evals:
        # Check required fields
        assert "decision" in res
        assert "action" in res
        assert "alert" in res
        assert "p_deny" in res
        assert "p_alert" in res
        assert 0.0 <= res["p_deny"] <= 1.0
        assert 0.0 <= res["p_alert"] <= 1.0

        # Enforce domain invariant: blocking strictly implies alerting
        if res["action"] == "deny" or res["p_deny"] >= 0.50:
            assert res["alert"] is True
            assert res["decision"] == "DENY (403)"


def test_sample_traffic_benign_webhooks_allowed(sample_traffic_lines, trained_surrogate):
    model, tokenizer = trained_surrogate
    evals = inspect_log_stream(sample_traffic_lines, model=model, tokenizer=tokenizer)

    # Check that legitimate signed Home Assistant webhooks are ALLOWED
    webhook_evals = [e for e in evals if "/webhook.php" in e["state"].get("uri", "")]
    assert len(webhook_evals) >= 2, "Expected /webhook.php log entries in sample traffic"
    for item in webhook_evals:
        assert item["action"] == "allow", f"Expected /webhook.php to be allowed: {item}"
        assert item["decision"] == "ALLOW (200)"
        assert item["p_deny"] < 0.20


def test_sample_traffic_known_probes_denied(sample_traffic_lines, trained_surrogate):
    model, tokenizer = trained_surrogate
    evals = inspect_log_stream(sample_traffic_lines, model=model, tokenizer=tokenizer)

    # Verify active scanners and sensitive file probes are DENIED
    probes = ["/phpinfo.php", "/xmlrpc.php?rsd", "wlwmanifest.xml", "/mcp"]
    for probe in probes:
        matching = [e for e in evals if probe in e["state"].get("uri", "")]
        assert len(matching) > 0, f"Expected probe {probe} in sample_traffic.log"
        for item in matching:
            assert item["decision"] == "DENY (403)", f"Expected {probe} to be denied: {item}"
            assert item["p_deny"] >= 0.50
            assert item["alert"] is True
