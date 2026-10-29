"""Rules engine for classifying parsed access logs into WAF training labels."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import yaml


def normalize_action(action: str) -> str:
    """Normalizes action names to 'allow' or 'deny'."""
    a = (action or "allow").strip().lower()
    if a in ("block", "deny", "reject", "drop"):
        return "deny"
    return "allow"


class ClassificationResult(BaseModel):
    """Result of classifying a request log entry."""
    action: str = Field(
        ...,
        description="'allow' or 'deny'",
    )
    alert: bool = Field(..., description="Whether this request should trigger a security alert (true/false)")
    is_malicious: str = Field(default="false", description="'true' or 'false' string for SystemOne model")
    severity_score: int = Field(default=0, ge=0, le=4, description="Severity score from 0 (benign) to 4 (critical)")
    primary_attack_vector: str = Field(
        default="benign",
        description="Attack category ('benign', 'scanner_recon', 'cmd_path_lfi', 'sqli', etc.)",
    )
    rule_id: Optional[str] = Field(default=None, description="Identifier of the rule that matched")
    reason: Optional[str] = Field(default=None, description="Human-readable explanation of the classification")

    def model_post_init(self, __context: Any) -> None:
        """Enforces the invariant: Any denied request strictly requires alerting."""
        self.action = normalize_action(self.action)
        if self.action == "deny":
            self.alert = True


class RuleCondition(BaseModel):
    """Criteria for matching an HTTP request."""
    status: Optional[Union[int, List[int]]] = None
    method: Optional[Union[str, List[str]]] = None
    path_regex: Optional[str] = None
    path_exact: Optional[str] = None
    path_prefix: Optional[str] = None
    user_agent_regex: Optional[str] = None
    user_agent_contains: Optional[str] = None
    protocol: Optional[str] = None


class Rule(BaseModel):
    """A classification rule."""
    id: str
    description: Optional[str] = None
    condition: RuleCondition
    action: str = "block"
    alert: bool = True
    is_malicious: str = "true"
    severity_score: int = 2
    primary_attack_vector: str = "scanner_recon"
    reason: Optional[str] = None

    def matches(
        self,
        status_code: int,
        method: str,
        path: str,
        user_agent: Optional[str],
        protocol: Optional[str],
    ) -> bool:
        """Evaluates whether this rule matches the request parameters."""
        c = self.condition

        # 1. Status match
        if c.status is not None:
            if isinstance(c.status, list):
                if status_code not in c.status:
                    return False
            elif status_code != c.status:
                return False

        # 2. Method match
        if c.method is not None:
            if isinstance(c.method, list):
                if method.upper() not in [m.upper() for m in c.method]:
                    return False
            elif method.upper() != c.method.upper():
                return False

        # 3. Path matches
        if c.path_exact is not None and path != c.path_exact:
            return False
        if c.path_prefix is not None and not path.startswith(c.path_prefix):
            return False
        if c.path_regex is not None:
            if not re.search(c.path_regex, path, re.IGNORECASE):
                return False

        # 4. User Agent matches
        ua = user_agent or ""
        if c.user_agent_contains is not None:
            if c.user_agent_contains.lower() not in ua.lower():
                return False
        if c.user_agent_regex is not None:
            if not re.search(c.user_agent_regex, ua, re.IGNORECASE):
                return False

        # 5. Protocol match
        if c.protocol is not None:
            proto = protocol or ""
            if c.protocol.upper() != proto.upper():
                return False

        return True

    def to_result(self) -> ClassificationResult:
        act = normalize_action(self.action)
        return ClassificationResult(
            action=act,
            alert=True if act == "deny" else self.alert,
            is_malicious=self.is_malicious,
            severity_score=self.severity_score,
            primary_attack_vector=self.primary_attack_vector,
            rule_id=self.id,
            reason=self.reason or self.description or self.id,
        )


# Default status code mappings reflecting core requirements:
# - 429 or 500: alerted and blocked
# - 404: alerted and probably blocked (blocked)
# - 200: good (allowed, no alert)
DEFAULT_STATUS_RULES: Dict[int, Dict[str, Any]] = {
    200: {
        "action": "allow",
        "alert": False,
        "is_malicious": "false",
        "severity_score": 0,
        "primary_attack_vector": "benign",
        "reason": "HTTP 200 OK - Benign traffic",
    },
    204: {
        "action": "allow",
        "alert": False,
        "is_malicious": "false",
        "severity_score": 0,
        "primary_attack_vector": "benign",
        "reason": "HTTP 204 No Content - Benign traffic",
    },
    206: {
        "action": "allow",
        "alert": False,
        "is_malicious": "false",
        "severity_score": 0,
        "primary_attack_vector": "benign",
        "reason": "HTTP 206 Partial Content - Benign traffic",
    },
    301: {
        "action": "block",
        "alert": True,
        "is_malicious": "true",
        "severity_score": 2,
        "primary_attack_vector": "scanner_recon",
        "reason": "HTTP 301 Moved Permanently - Bot redirected to tarpit",
    },
    302: {
        "action": "block",
        "alert": True,
        "is_malicious": "true",
        "severity_score": 2,
        "primary_attack_vector": "scanner_recon",
        "reason": "HTTP 302 Found - Bot redirected to tarpit",
    },
    304: {
        "action": "allow",
        "alert": False,
        "is_malicious": "false",
        "severity_score": 0,
        "primary_attack_vector": "benign",
        "reason": "HTTP 304 Not Modified - Cache validation",
    },
    308: {
        "action": "block",
        "alert": True,
        "is_malicious": "true",
        "severity_score": 2,
        "primary_attack_vector": "scanner_recon",
        "reason": "HTTP 308 Permanent Redirect - Bot redirected to tarpit",
    },
    400: {
        "action": "block",
        "alert": True,
        "is_malicious": "true",
        "severity_score": 2,
        "primary_attack_vector": "scanner_recon",
        "reason": "HTTP 400 Bad Request - Malformed request or probe",
    },
    403: {
        "action": "block",
        "alert": True,
        "is_malicious": "true",
        "severity_score": 2,
        "primary_attack_vector": "scanner_recon",
        "reason": "HTTP 403 Forbidden - Policy violation or unauthorized probe",
    },
    404: {
        "action": "block",
        "alert": True,
        "is_malicious": "true",
        "severity_score": 2,
        "primary_attack_vector": "scanner_recon",
        "reason": "HTTP 404 Not Found - Alerted and blocked reconnaissance probe",
    },
    408: {
        "action": "block",
        "alert": True,
        "is_malicious": "true",
        "severity_score": 1,
        "primary_attack_vector": "scanner_recon",
        "reason": "HTTP 408 Request Timeout - Slowloris or incomplete probe",
    },
    418: {
        "action": "block",
        "alert": True,
        "is_malicious": "true",
        "severity_score": 2,
        "primary_attack_vector": "scanner_recon",
        "reason": "HTTP 418 Honeypot Teapot - Blocked scanner",
    },
    421: {
        "action": "block",
        "alert": True,
        "is_malicious": "true",
        "severity_score": 2,
        "primary_attack_vector": "scanner_recon",
        "reason": "HTTP 421 Misdirected Request",
    },
    429: {
        "action": "block",
        "alert": True,
        "is_malicious": "true",
        "severity_score": 3,
        "primary_attack_vector": "rate_limit_or_challenge",
        "reason": "HTTP 429 Too Many Requests - Alerted and blocked rate limit flood",
    },
    500: {
        "action": "block",
        "alert": True,
        "is_malicious": "true",
        "severity_score": 3,
        "primary_attack_vector": "server_error_anomaly",
        "reason": "HTTP 500 Internal Server Error - Alerted and blocked server failure",
    },
}


class FolderRules:
    """Manages classification rules for a specific folder or service."""

    def __init__(
        self,
        name: str = "default",
        description: str = "",
        parser_format: Optional[str] = None,
        status_defaults: Optional[Dict[int, Dict[str, Any]]] = None,
        rules: Optional[List[Rule]] = None,
    ):
        self.name = name
        self.description = description
        self.parser_format = parser_format
        self.status_defaults: Dict[int, Dict[str, Any]] = dict(DEFAULT_STATUS_RULES)
        if status_defaults:
            self.status_defaults.update(status_defaults)
        self.rules: List[Rule] = rules or []

    def classify(
        self,
        status_code: int,
        method: str,
        path: str,
        user_agent: Optional[str] = None,
        protocol: Optional[str] = None,
    ) -> ClassificationResult:
        """
        Classifies an HTTP request based on custom rules first, then status defaults.
        """
        # 1. Evaluate explicit custom rules in priority order
        for rule in self.rules:
            if rule.matches(status_code, method, path, user_agent, protocol):
                return rule.to_result()

        # 2. Status code default rule
        if status_code in self.status_defaults:
            spec = self.status_defaults[status_code]
            return ClassificationResult(
                action=spec["action"],
                alert=spec["alert"],
                is_malicious=spec["is_malicious"],
                severity_score=spec["severity_score"],
                primary_attack_vector=spec["primary_attack_vector"],
                rule_id=f"default_status_{status_code}",
                reason=spec.get("reason", f"Status code {status_code}"),
            )

        # 3. Fallback for unrecognized status code:
        # If 4xx or 5xx, alert and deny; if 2xx or 3xx, allow
        if 400 <= status_code < 600:
            return ClassificationResult(
                action="deny",
                alert=True,
                is_malicious="true",
                severity_score=2,
                primary_attack_vector="scanner_recon",
                rule_id="default_fallback_error",
                reason=f"HTTP {status_code} error response - Alerted and denied",
            )
        else:
            return ClassificationResult(
                action="allow",
                alert=False,
                is_malicious="false",
                severity_score=0,
                primary_attack_vector="benign",
                rule_id="default_fallback_success",
                reason=f"HTTP {status_code} - Allowed",
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], name: Optional[str] = None) -> "FolderRules":
        """Builds FolderRules from parsed dictionary."""
        rule_name = data.get("name") or data.get("service_name") or name or "default"
        description = data.get("description", "")
        parser_cfg = data.get("parser")
        if isinstance(parser_cfg, dict):
            parser_format = parser_cfg.get("format")
        elif isinstance(parser_cfg, str):
            parser_format = parser_cfg
        else:
            parser_format = data.get("format")

        status_defs = {}
        raw_defs = data.get("status_defaults", {})
        for k, v in raw_defs.items():
            try:
                status_defs[int(k)] = v
            except (ValueError, TypeError):
                pass

        parsed_rules: List[Rule] = []
        for r in data.get("rules", []):
            parsed_rules.append(Rule(**r))

        return cls(
            name=rule_name,
            description=description,
            parser_format=parser_format,
            status_defaults=status_defs,
            rules=parsed_rules,
        )

    @classmethod
    def from_yaml_file(cls, filepath: Union[str, Path]) -> "FolderRules":
        """Loads folder rules from a YAML file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data, name=filepath.parent.name)


def load_rules_for_folder(folder_path: Union[str, Path]) -> FolderRules:
    """
    Finds and loads rules from rules.yaml, rules.yml, or rules.json in the folder.
    If no rule file is found, returns default FolderRules for that folder.
    """
    folder = Path(folder_path)
    for filename in ["rules.yaml", "rules.yml", "rules.json"]:
        cand = folder / filename
        if cand.is_file():
            return FolderRules.from_yaml_file(cand)

    # Return default rules matching core requirements
    return FolderRules(name=folder.name, description=f"Default rules for {folder.name}")
