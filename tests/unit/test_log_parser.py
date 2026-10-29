"""Unit tests for access log parsing and pre-response sanitization."""

import pytest

from wafer.data.log_parser import (
    parse_caddy_log_line,
    parse_apache_log_line,
    parse_and_sanitize_log_line,
    logs_to_waf_records,
)
from wafer.model.backbone import MockTokenizer
from wafer.model.encoder import encode_record


CADDY_LINE = (
    '2026/10/28 12:12:47.053\tINFO\thttp.log.access\thandled request\t'
    '{"request": {"remote_ip": "2001:db8:1::10", '
    '"client_ip": "2001:db8:1::10", "proto": "HTTP/2.0", '
    '"method": "GET", "host": "ha.example.com", '
    '"uri": "/api/hassio/addons/e1b5c59f_caddy-custom/logs/follow?lines=100"}, '
    '"duration": 0, "size": 41496, "status": 200}'
)

APACHE_LINE = (
    '2001:db8:2::20 - - [28/Oct/2026:04:45:02 -0700] '
    '"GET /webhook.php?timestamp=1793187900&signature=1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff&ipv4=198.51.100.61&ipv6=2001:db8:2::20 HTTP/1.1" '
    '200 5309 "-" "Home-Assistant-Webhook"'
)


def test_parse_caddy_sanitization():
    state = parse_caddy_log_line(CADDY_LINE)
    assert state is not None

    # Verify extracted request attributes
    assert state.method == "GET"
    assert state.uri == "/api/hassio/addons/e1b5c59f_caddy-custom/logs/follow?lines=100"
    assert state.client_ip_asn == "2001:db8:1::10"
    assert state.headers.get("Host") == "ha.example.com"

    # Verify response attributes are completely sanitized out
    state_dict = state.model_dump()
    assert "status" not in state_dict
    assert "duration" not in state_dict
    assert "size" not in state_dict


def test_parse_apache_sanitization():
    state = parse_apache_log_line(APACHE_LINE)
    assert state is not None

    assert state.method == "GET"
    assert "/webhook.php" in state.uri
    assert state.client_ip_asn == "2001:db8:2::20"
    assert state.headers.get("User-Agent") == "Home-Assistant-Webhook"

    state_dict = state.model_dump()
    assert "status" not in state_dict
    assert "size" not in state_dict


def test_logs_to_waf_records_packing():
    multiline = f"{CADDY_LINE}\n{APACHE_LINE}\n"
    records = logs_to_waf_records(multiline)

    assert len(records) == 2
    tokenizer = MockTokenizer()

    for r in records:
        enc = encode_record(tokenizer, r)
        assert enc.input_ids.size(0) > 0
        assert len(enc.option_spans) == 2  # Standard 2 questions (action, alert)
