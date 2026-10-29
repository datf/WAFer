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
    model_config = {"extra": "ignore"}

    action: str = Field(
        ...,
        description="'allow' or 'deny'",
    )
    alert: bool = Field(..., description="Whether this request should trigger a security alert (true/false)")
    rule_id: Optional[str] = Field(default=None, description="Identifier of the rule that matched")

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
    model_config = {"extra": "ignore"}

    id: str
    description: Optional[str] = None
    condition: RuleCondition
    action: str = "deny"
    alert: bool = True

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
            rule_id=self.id,
        )


# Default status code mappings reflecting core requirements:
# - 429 or 500: alerted and denied
# - 404: alerted and denied
# - 301, 302, 308 (redirects/tarpits): alerted and denied
# - 200, 204, 206, 304: good (allowed, no alert)
DEFAULT_STATUS_RULES: Dict[int, Dict[str, Any]] = {
    200: {"action": "allow", "alert": False},
    204: {"action": "allow", "alert": False},
    206: {"action": "allow", "alert": False},
    301: {"action": "deny", "alert": True},
    302: {"action": "deny", "alert": True},
    304: {"action": "allow", "alert": False},
    308: {"action": "deny", "alert": True},
    400: {"action": "deny", "alert": True},
    403: {"action": "deny", "alert": True},
    404: {"action": "deny", "alert": True},
    408: {"action": "deny", "alert": True},
    418: {"action": "deny", "alert": True},
    421: {"action": "deny", "alert": True},
    429: {"action": "deny", "alert": True},
    500: {"action": "deny", "alert": True},
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
            act = normalize_action(spec.get("action", "allow"))
            alt = bool(spec.get("alert", False))
            return ClassificationResult(
                action=act,
                alert=alt,
                rule_id=f"default_status_{status_code}",
            )

        # 3. Fallback for unrecognized status code:
        # If 4xx or 5xx, alert and deny; if 2xx or 3xx, allow
        if 400 <= status_code < 600:
            return ClassificationResult(
                action="deny",
                alert=True,
                rule_id="default_fallback_error",
            )
        else:
            return ClassificationResult(
                action="allow",
                alert=False,
                rule_id="default_fallback_success",
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
