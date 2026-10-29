"""Unit tests for the WAF classification rules engine."""

from pathlib import Path
import pytest

from wafer.data.rules import (
    ClassificationResult,
    FolderRules,
    Rule,
    RuleCondition,
    load_rules_for_folder,
)


def test_default_status_rules():
    """Verify core status code default requirements."""
    rules = FolderRules(name="test")

    # 1. 200 should be good (allowed, alert=False)
    r200 = rules.classify(status_code=200, method="GET", path="/")
    assert r200.action == "allow"
    assert not r200.alert
    assert r200.is_malicious == "false"
    assert r200.severity_score == 0

    # 2. 404 should be alerted and denied
    r404 = rules.classify(status_code=404, method="GET", path="/missing")
    assert r404.action == "deny"
    assert r404.alert is True

    # 3. 429 should be alerted and denied
    r429 = rules.classify(status_code=429, method="POST", path="/trace/")
    assert r429.action == "deny"
    assert r429.alert is True

    # 4. 500 should be alerted and denied
    r500 = rules.classify(status_code=500, method="GET", path="/sitemap.xml")
    assert r500.action == "deny"
    assert r500.alert is True

    # 5. 301, 302, 308 (tarpit redirects) should be alerted and denied
    for code in (301, 302, 308):
        r_redirect = rules.classify(status_code=code, method="GET", path="/bot-target")
        assert r_redirect.action == "deny"
        assert r_redirect.alert is True
        assert r_redirect.is_malicious == "true"


def test_custom_rule_matching():
    """Test explicit custom rule overrides on path, method, and user agent."""
    rule = Rule(
        id="webhook_allow",
        condition=RuleCondition(
            method="POST",
            path_prefix="/webhook/",
            user_agent_contains="Home-Assistant",
            status=200,
        ),
        action="allow",
        alert=False,
        is_malicious="false",
        severity_score=0,
    )
    rules = FolderRules(name="custom", rules=[rule])

    # Matching request
    res = rules.classify(
        status_code=200,
        method="POST",
        path="/webhook/sensor",
        user_agent="Home-Assistant-Webhook/2026.1",
    )
    assert res.rule_id == "webhook_allow"
    assert res.action == "allow"
    assert not res.alert

    # Non-matching user agent falls back to status default
    res2 = rules.classify(
        status_code=404,
        method="POST",
        path="/webhook/sensor",
        user_agent="MaliciousBot/1.0",
    )
    assert res2.rule_id != "webhook_allow"
    assert res2.action == "deny"
    assert res2.alert is True


def test_folder_rules_loading(tmp_path):
    """Verify loading and applying rules for service directories."""
    # Test default rules fallback
    rules = load_rules_for_folder(tmp_path)
    assert rules.name == tmp_path.name

    # Assert core constraints on rules classification
    # 200 OK
    res_200 = rules.classify(200, "GET", "/", "Mozilla/5.0")
    assert res_200.action == "allow"
    assert not res_200.alert

    # 404 Not Found
    res_404 = rules.classify(404, "GET", "/non_existent_page_12345", "Mozilla/5.0")
    assert res_404.action == "deny"
    assert res_404.alert is True

    # 429 Rate Limit
    res_429 = rules.classify(429, "GET", "/", "Mozilla/5.0")
    assert res_429.action == "deny"
    assert res_429.alert is True

    # 500 Server Error
    res_500 = rules.classify(500, "GET", "/", "Mozilla/5.0")
    assert res_500.action == "deny"
    assert res_500.alert is True
