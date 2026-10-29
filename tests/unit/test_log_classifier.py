"""Unit tests for the log parser and classifier pipeline."""

import gzip
from pathlib import Path
import pytest

from wafer.data.log_parser import (
    ParsedLogEntry,
    build_waf_training_record,
    find_log_files_in_folder,
    open_log_file,
    parse_clf_log_line,
    parse_caddy_log_entry,
    parse_log_entry,
    split_request_line,
)
from wafer.data.log_classifier import process_folder, resolve_log_dir
from wafer.data.rules import load_rules_for_folder


SAMPLE_LINE_200 = (
    '85.85.152.14 - - [28/Oct/2026:00:17:01 -0700] '
    '"POST / HTTP/2.0" 200 114 "-" '
    '"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36"'
)

SAMPLE_LINE_404 = (
    '2a14:c380:1300:ef::a - - [28/Oct/2026:01:21:28 -0700] '
    '"GET /inputs.php HTTP/1.1" 404 516 "-" "Go-http-client/1.1"'
)

SAMPLE_LINE_429 = (
    '54.87.11.104 - - [28/Oct/2026:01:17:13 -0700] '
    '"POST /trace/ HTTP/1.1" 429 5269 "https://app.example.com/" '
    '"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"'
)

SAMPLE_LINE_500 = (
    '1.2.3.4 - - [28/Oct/2026:02:10:00 -0700] '
    '"GET /sitemap.xml HTTP/1.1" 500 245 "-" "Googlebot/2.1"'
)


def test_parse_clf_extraction():
    """Verify extraction of method, path, protocol, user agent, IP, status."""
    entry = parse_clf_log_line(SAMPLE_LINE_200)
    assert entry is not None
    assert entry.method == "POST"
    assert entry.path == "/"
    assert entry.protocol == "HTTP/2.0"
    assert "Mozilla/5.0" in (entry.user_agent or "")
    assert entry.remote_ip == "85.85.152.14"
    assert entry.status_code == 200


def test_pre_response_sanitization():
    """
    Verify that incoming request state contains method, uri, protocol, user_agent
    and strictly EXCLUDES all response-level information (status, size, duration).
    """
    entry = parse_clf_log_line(SAMPLE_LINE_429)
    assert entry is not None

    state = entry.to_http_request_state()
    state_dict = state.model_dump()

    # Required extracted fields
    assert state_dict["method"] == "POST"
    assert state_dict["uri"] == "/trace/"
    assert state_dict["protocol"] == "HTTP/1.1"
    assert "Mozilla/5.0" in state_dict["headers"]["User-Agent"]
    assert state_dict["client_ip_asn"] == "54.87.11.104"

    # Strictly no response fields
    assert "status" not in state_dict
    assert "status_code" not in state_dict
    assert "size" not in state_dict
    assert "duration" not in state_dict


def test_split_request_line_edge_cases():
    """Test robust handling of unusual or scanner-crafted request lines."""
    # Standard
    assert split_request_line("GET /index.html HTTP/1.1") == ("GET", "/index.html", "HTTP/1.1")
    assert split_request_line("OPTIONS * RTSP/1.0") == ("OPTIONS", "*", "RTSP/1.0")

    # Missing protocol
    assert split_request_line("GET /simple") == ("GET", "/simple", "HTTP/1.0")

    # Scanner / probe junk
    assert split_request_line("LEAKIX") == ("LEAKIX", "/", "UNKNOWN")
    assert split_request_line("\\x16\\x03") == ("\\x16\\x03", "/", "UNKNOWN")
    assert split_request_line("-") == ("UNKNOWN", "/", "HTTP/1.1")
    assert split_request_line("fox a 1 -1 fox hello\n") == ("fox", "a 1 -1 fox hello", "UNKNOWN")


def test_transparent_gzip_reading(tmp_path: Path):
    """Test that open_log_file transparently reads both .gz and plain text files."""
    plain_file = tmp_path / "access.log"
    gz_file = tmp_path / "access.log.gz"

    content = f"{SAMPLE_LINE_200}\n{SAMPLE_LINE_404}\n"
    plain_file.write_text(content, encoding="utf-8")

    with gzip.open(gz_file, "wt", encoding="utf-8") as f:
        f.write(content)

    # Read plain text
    lines_plain = list(open_log_file(plain_file))
    assert len(lines_plain) == 2
    assert "POST / HTTP/2.0" in lines_plain[0]

    # Read gzip
    lines_gz = list(open_log_file(gz_file))
    assert len(lines_gz) == 2
    assert "POST / HTTP/2.0" in lines_gz[0]


def test_training_record_structure():
    """Verify that build_waf_training_record creates complete training records."""
    entry = parse_clf_log_line(SAMPLE_LINE_500)
    assert entry is not None

    rules = load_rules_for_folder(Path("."))
    res = rules.classify(entry.status_code, entry.method, entry.path, entry.user_agent, entry.protocol)
    record = build_waf_training_record(entry, res)

    # Check state (simplified inputs)
    assert record["state"]["method"] == "GET"
    assert record["state"]["uri"] == "/sitemap.xml"
    assert record["state"]["protocol"] == "HTTP/1.1"
    assert record["state"]["user_agent"] == "Googlebot/2.1"
    assert "status" not in record["state"]

    # Check targets for 500 error: alerted and denied
    assert record["targets"]["action"] == "deny"
    assert record["targets"]["alert"] == "true"

    # Check questions present
    assert "action" in record["questions"]
    assert "alert" in record["questions"]


def test_process_sample_log_folder(tmp_path):
    """Verify parsing and classifying log files from a directory."""
    sample_log = tmp_path / "sample.log"
    sample_log.write_text(SAMPLE_LINE_200 + "\n" + SAMPLE_LINE_404 + "\n" + SAMPLE_LINE_500 + "\n")

    stats = process_folder(tmp_path, limit=100)
    assert stats["parsed_lines"] == 3
    assert stats["failed_lines"] == 0
    assert stats["actions"]["allow"] > 0
    assert len(stats["validation_errors"]) == 0


def test_parse_caddy_log_entry():
    """Verify structured parsing of Caddy JSON access log lines."""
    caddy_line = (
        '2026/08/26 16:10:30.030\tINFO\thttp.log.access\thandled request\t'
        '{"request": {"remote_ip": "108.177.64.70", "client_ip": "108.177.64.70", '
        '"proto": "HTTP/1.1", "method": "POST", "host": "ha.example.com", '
        '"uri": "/auth/token", "headers": {"User-Agent": ["Home-Assistant-App/1.0"]}}, '
        '"duration": 0.0805, "size": 242, "status": 200}'
    )
    entry = parse_caddy_log_entry(caddy_line)
    assert entry is not None
    assert entry.method == "POST"
    assert entry.path == "/auth/token"
    assert entry.protocol == "HTTP/1.1"
    assert entry.status_code == 200
    assert entry.user_agent == "Home-Assistant-App/1.0"
    assert entry.host == "ha.example.com"

    # Verify pre-response sanitization
    state = entry.to_http_request_state()
    state_dict = state.model_dump()
    assert state_dict["method"] == "POST"
    assert state_dict["uri"] == "/auth/token"
    assert state_dict["protocol"] == "HTTP/1.1"
    assert "status" not in state_dict
    assert "duration" not in state_dict
    assert "size" not in state_dict


def test_parse_log_entry_autodetect():
    """Verify auto-detection between Caddy JSON format and Apache CLF."""
    clf_line = SAMPLE_LINE_200
    caddy_line = (
        '2026/08/26 16:10:30.030\tINFO\thttp.log.access\thandled request\t'
        '{"request": {"remote_ip": "1.2.3.4", "proto": "HTTP/2.0", "method": "GET", '
        '"uri": "/"}, "status": 200}'
    )

    entry1 = parse_log_entry(clf_line)
    assert entry1 is not None
    assert entry1.protocol == "HTTP/2.0"
    assert entry1.method == "POST"

    entry2 = parse_log_entry(caddy_line)
    assert entry2 is not None
    assert entry2.protocol == "HTTP/2.0"
    assert entry2.method == "GET"

