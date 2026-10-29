"""Log parser and request state sanitizer for Caddy and Apache access logs."""

import gzip
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from wafer.data.schema import HTTPRequestState
from wafer.data.synthetic import STANDARD_QUESTIONS
from wafer.data.rules import ClassificationResult, FolderRules, load_rules_for_folder


# Regex matching Apache Combined Log Format:
# remote_ip - - [timestamp] "REQUEST_STRING" status size "referer" "user_agent"
CLF_REGEX = re.compile(
    r'^(?P<remote_ip>[\w.:]+)\s+-\s+-\s+\[(?P<timestamp>[^\]]+)\]\s+"(?P<request>[^"]*)"\s+(?P<status>\d+)\s+(?P<size>\S+)(?:\s+"(?P<referer>[^"]*)"\s+"(?P<user_agent>[^"]*)")?'
)


class ParsedLogEntry(BaseModel):
    """Structured representation of a parsed access log entry."""
    remote_ip: str = Field(..., description="Client IP address")
    timestamp: str = Field(..., description="Access timestamp")
    method: str = Field(..., description="HTTP Method (GET, POST, etc.)")
    path: str = Field(..., description="Request path and query string")
    protocol: str = Field(..., description="HTTP Protocol (HTTP/1.1, HTTP/2.0, etc.)")
    status_code: int = Field(..., description="HTTP Response Status Code")
    size_bytes: int = Field(default=0, description="Response size in bytes")
    host: Optional[str] = Field(default=None, description="HTTP Host header")
    referer: Optional[str] = Field(default=None, description="HTTP Referer header")
    user_agent: Optional[str] = Field(default=None, description="HTTP User-Agent header")
    raw_line: str = Field(default="", description="Original raw log line")

    def to_http_request_state(self) -> HTTPRequestState:
        """
        Converts log entry to pre-response HTTPRequestState.
        Explicitly SANITIZES and DROPS response attributes (status_code, size_bytes).
        """
        headers: Dict[str, str] = {}
        if self.host:
            headers["Host"] = self.host
        if self.user_agent and self.user_agent != "-":
            headers["User-Agent"] = self.user_agent
        if self.referer and self.referer != "-":
            headers["Referer"] = self.referer

        return HTTPRequestState(
            method=self.method,
            uri=self.path,
            protocol=self.protocol,
            user_agent=self.user_agent if self.user_agent != "-" else None,
            client_ip_asn=self.remote_ip,
            headers=headers,
            body=None,
        )


def split_request_line(req_str: str) -> Tuple[str, str, str]:
    """
    Robustly parses the HTTP request line into (method, path, protocol).
    Handles standard lines ('GET /path HTTP/1.1'), two-token lines ('GET /path'),
    empty or token-only scanner probes ('LEAKIX', '\x16\x03', '-').
    """
    req_str = req_str.strip()
    if not req_str or req_str == "-":
        return "UNKNOWN", "/", "HTTP/1.1"

    parts = req_str.split()
    if len(parts) == 1:
        return parts[0], "/", "UNKNOWN"
    elif len(parts) == 2:
        return parts[0], parts[1], "HTTP/1.0"
    elif len(parts) == 3:
        return parts[0], parts[1], parts[2]
    else:
        # Check if the last token is a protocol (e.g. HTTP/1.1, RTSP/1.0)
        if parts[-1].startswith("HTTP/") or "/" in parts[-1]:
            return parts[0], " ".join(parts[1:-1]), parts[-1]
        return parts[0], " ".join(parts[1:]), "UNKNOWN"


def parse_clf_log_line(line: str) -> Optional[ParsedLogEntry]:
    """
    Parses a single Combined Log Format (Apache) line into a ParsedLogEntry.
    Returns None if line does not match or is empty.
    """
    line = line.strip()
    if not line:
        return None

    match = CLF_REGEX.match(line)
    if not match:
        return None

    gd = match.groupdict()
    remote_ip = gd.get("remote_ip", "0.0.0.0")
    timestamp = gd.get("timestamp", "")
    req_str = gd.get("request", "")
    status_str = gd.get("status", "0")
    size_str = gd.get("size", "0")
    referer = gd.get("referer")
    user_agent = gd.get("user_agent")

    method, path, protocol = split_request_line(req_str)

    try:
        status_code = int(status_str)
    except (ValueError, TypeError):
        status_code = 0

    try:
        size_bytes = int(size_str)
    except (ValueError, TypeError):
        size_bytes = 0

    if user_agent == "-":
        user_agent = None
    if referer == "-":
        referer = None

    return ParsedLogEntry(
        remote_ip=remote_ip,
        timestamp=timestamp,
        method=method,
        path=path,
        protocol=protocol,
        status_code=status_code,
        size_bytes=size_bytes,
        referer=referer,
        user_agent=user_agent,
        raw_line=line,
    )


def parse_caddy_log_entry(line: str) -> Optional[ParsedLogEntry]:
    """
    Parses a Caddy access log line (JSON payload) into a ParsedLogEntry.
    """
    line = line.strip()
    if not line:
        return None

    json_start = line.find("{")
    if json_start == -1:
        return None

    try:
        data = json.loads(line[json_start:])
    except (json.JSONDecodeError, ValueError):
        return None

    req = data.get("request", {})
    if not req:
        return None

    remote_ip = req.get("client_ip") or req.get("remote_ip") or "0.0.0.0"
    method = req.get("method", "GET")
    uri = req.get("uri", "/")
    proto = req.get("proto", "HTTP/1.1")
    host = req.get("host")

    # Timestamp from line prefix if available
    prefix_parts = line[:json_start].split()
    timestamp = prefix_parts[0] if prefix_parts else ""

    status_code = data.get("status", 0)
    size_bytes = data.get("size", 0)

    # Extract user agent and referer if available in headers
    headers = req.get("headers", {})
    user_agent = None
    referer = None
    if isinstance(headers, dict):
        for k, v in headers.items():
            k_lower = k.lower()
            val = v[0] if isinstance(v, list) and v else (v if isinstance(v, str) else None)
            if k_lower == "user-agent":
                user_agent = val
            elif k_lower == "referer":
                referer = val

    return ParsedLogEntry(
        remote_ip=remote_ip,
        timestamp=timestamp,
        method=method,
        path=uri,
        protocol=proto,
        status_code=status_code,
        size_bytes=size_bytes,
        host=host,
        referer=referer,
        user_agent=user_agent,
        raw_line=line,
    )


def parse_log_entry(line: str, format_hint: Optional[str] = None) -> Optional[ParsedLogEntry]:
    """
    Parses an access log line using the specified format hint or auto-detection.
    Supports Caddy JSON logs and Apache Combined Log Format.
    """
    line = line.strip()
    if not line:
        return None

    hint = (format_hint or "").lower()
    if hint in ("caddy", "json"):
        entry = parse_caddy_log_entry(line)
        if entry is not None:
            return entry
        return parse_clf_log_line(line)

    if hint in ("apache", "clf", "combined"):
        entry = parse_clf_log_line(line)
        if entry is not None:
            return entry
        return parse_caddy_log_entry(line)

    # Auto-detection:
    if "handled request" in line or (line.startswith("{") and '"request"' in line) or ('"proto"' in line and '"status"' in line):
        entry = parse_caddy_log_entry(line)
        if entry is not None:
            return entry

    return parse_clf_log_line(line)


def is_gzip_file(filepath: Union[str, Path]) -> bool:
    """Checks whether a file is gzip compressed via suffix or magic bytes."""
    path = Path(filepath)
    if path.suffix == ".gz":
        return True
    try:
        with open(path, "rb") as f:
            magic = f.read(2)
            return magic == b"\x1f\x8b"
    except (OSError, IOError):
        return False


def open_log_file(filepath: Union[str, Path]) -> Iterator[str]:
    """
    Yields lines from a log file transparently handling plain text and gzip.
    """
    path = Path(filepath)
    if is_gzip_file(path):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line
    else:
        with open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line


def find_log_files_in_folder(folder_path: Union[str, Path]) -> List[Path]:
    """
    Finds all access log files in a directory, ignoring rule files, hidden files,
    and non-log files.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        return []

    log_files: List[Path] = []
    ignored_exts = {".yaml", ".yml", ".json", ".md", ".py", ".pyc", ".sh", ".txt"}
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.startswith("."):
                continue
            p = Path(root) / f
            # Match standard access.log naming conventions
            if "access.log" in f or "access" in f:
                log_files.append(p)
            elif p.suffix.lower() not in ignored_exts:
                log_files.append(p)

    return sorted(log_files)


def build_waf_training_record(
    entry: ParsedLogEntry,
    classification: ClassificationResult,
    questions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Constructs a WAF training record containing:
    1. HTTP request state (strictly simplified: method, uri, protocol, user_agent).
    2. Questions dictionary (STANDARD_QUESTIONS: action and alert).
    3. Ground truth targets (action: 'allow'|'deny', alert: 'true'|'false').
    4. Diagnostic classification metadata.
    """
    q_specs = questions or STANDARD_QUESTIONS
    state = entry.to_http_request_state()

    act = "deny" if classification.action in ("block", "deny") else "allow"
    alt = "true" if (classification.alert or act == "deny") else "false"

    return {
        "state": state.to_model_state(),
        "questions": q_specs,
        "targets": {
            "action": act,
            "alert": alt,
        },
        "classification": {
            "action": act,
            "alert": alt == "true",
            "status_code": entry.status_code,
            "rule_id": classification.rule_id,
        },
    }


def parse_and_classify_file(
    filepath: Union[str, Path],
    rules: Optional[FolderRules] = None,
    limit: Optional[int] = None,
) -> Iterator[Tuple[ParsedLogEntry, ClassificationResult, Dict[str, Any]]]:
    """
    Parses a log file line-by-line, applies classification rules,
    and yields (ParsedLogEntry, ClassificationResult, training_record).
    """
    path = Path(filepath)
    if rules is None:
        rules = load_rules_for_folder(path.parent)

    count = 0
    for line in open_log_file(path):
        entry = parse_log_entry(line, format_hint=rules.parser_format)
        if entry is None:
            continue

        classification = rules.classify(
            status_code=entry.status_code,
            method=entry.method,
            path=entry.path,
            user_agent=entry.user_agent,
            protocol=entry.protocol,
        )
        record = build_waf_training_record(entry, classification)
        yield entry, classification, record

        count += 1
        if limit is not None and count >= limit:
            break


# =========================================================================
# Legacy / Backwards Compatibility Functions
# =========================================================================

def parse_caddy_log_line(line: str) -> Optional[HTTPRequestState]:
    """
    Parses a Caddy access log line and extracts ONLY the incoming request state.
    Explicitly sanitizes and drops response-specific attributes: status, duration, size.
    """
    line = line.strip()
    if not line:
        return None

    json_start = line.find("{")
    if json_start == -1:
        return None

    try:
        data = json.loads(line[json_start:])
    except json.JSONDecodeError:
        return None

    req = data.get("request", {})
    if not req:
        return None

    method = req.get("method", "GET")
    uri = req.get("uri", "/")
    proto = req.get("proto")
    client_ip = req.get("client_ip") or req.get("remote_ip")
    host = req.get("host", "")

    headers: Dict[str, str] = {}
    if host:
        headers["Host"] = host

    req_headers = req.get("headers", {})
    user_agent = None
    if isinstance(req_headers, dict):
        for k, v in req_headers.items():
            val = v[0] if (isinstance(v, list) and v) else (v if isinstance(v, str) else "")
            headers[k] = val
            if k.lower() == "user-agent":
                user_agent = val

    return HTTPRequestState(
        method=method,
        uri=uri,
        protocol=proto,
        user_agent=user_agent,
        client_ip_asn=client_ip,
        headers=headers,
        body=None,
    )


def parse_apache_log_line(line: str) -> Optional[HTTPRequestState]:
    """
    Parses an Apache / Combined Log Format line and extracts ONLY the incoming request state.
    Explicitly sanitizes and drops response-specific attributes: status code, byte size.
    """
    entry = parse_clf_log_line(line)
    if entry is None:
        return None
    return entry.to_http_request_state()


def parse_and_sanitize_log_line(line: str) -> Optional[HTTPRequestState]:
    """Auto-detects format (Caddy JSON or Apache CLF) and returns sanitized request state."""
    line = line.strip()
    if not line:
        return None

    if "handled request" in line or (line.startswith("{") and '"request"' in line):
        return parse_caddy_log_line(line)

    clf_state = parse_apache_log_line(line)
    if clf_state is not None:
        return clf_state

    if "{" in line:
        return parse_caddy_log_line(line)

    return None


def logs_to_waf_records(
    log_text: str,
    questions: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Converts raw multiline access logs into a list of sanitized WAF records
    ready for model evaluation.
    """
    q_specs = questions or STANDARD_QUESTIONS
    records: List[Dict[str, Any]] = []

    for line in log_text.strip().splitlines():
        state = parse_and_sanitize_log_line(line)
        if state is not None:
            records.append({
                "state": state.to_model_state(),
                "questions": q_specs,
            })

    return records
