"""Pydantic schemas for SystemOne WAF records and HTTP states."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class HTTPRequestState(BaseModel):
    """Normalized HTTP request state."""
    method: str = Field(default="GET", description="HTTP request method (GET, POST, etc.)")
    uri: str = Field(default="/", description="Target path and query parameters")
    protocol: Optional[str] = Field(default=None, description="HTTP protocol version (e.g. HTTP/1.1, HTTP/2.0)")
    user_agent: Optional[str] = Field(default=None, description="Client User-Agent header string")
    client_ip_asn: Optional[str] = Field(default=None, description="IP address and autonomous system number")
    headers: Dict[str, str] = Field(default_factory=dict, description="Filtered HTTP headers")
    body: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Raw or parsed request body")

    def to_model_state(self) -> Dict[str, str]:
        """Returns the strictly simplified input dictionary for the WAF model."""
        return {
            "method": self.method,
            "uri": self.uri,
            "protocol": self.protocol or "HTTP/1.1",
            "user_agent": self.user_agent or self.headers.get("User-Agent", ""),
        }


class QuestionSpec(BaseModel):
    """Specification for a single SystemOne question."""
    type: str = Field(..., description="'noul' (boolean), 'choice' (categorical), or 'score' (ordinal)")
    instructions: Optional[str] = Field(default=None, description="Prompt instructions for the question")
    criteria: Optional[Union[Dict[str, str], List[str]]] = Field(
        default=None,
        description="Option key to description map (dict) or ordered scale descriptions (list)",
    )


class WAFRecord(BaseModel):
    """Full WAF record containing HTTP state, questions, and optional ground-truth targets."""
    state: Union[HTTPRequestState, Dict[str, Any], str] = Field(..., description="HTTP request state")
    questions: Dict[str, QuestionSpec] = Field(..., description="Dictionary of question ID to QuestionSpec")
    targets: Optional[Dict[str, Union[str, int, float, List[float]]]] = Field(
        default=None, description="Ground truth labels or soft probability vectors per question"
    )
