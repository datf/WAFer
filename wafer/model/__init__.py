"""Gemma-Clef model components and loss functions."""

from wafer.model.encoder import EncodedRecord, encode_record
from wafer.model.head import EvidenceRoutingLayer, CrossFieldAttentionLayer, JointSchemaHead
from wafer.model.backbone import (
    EmbeddingGemmaDecisionModel,
    GemmaClefDecisionModel,
    SurrogateDecisionModel,
    MODEL_ALIASES,
    resolve_model_id,
    get_backbone_output_dim,
)
from wafer.model.loss import compute_calibrated_systemone_loss

__all__ = [
    "EncodedRecord",
    "encode_record",
    "EvidenceRoutingLayer",
    "CrossFieldAttentionLayer",
    "JointSchemaHead",
    "EmbeddingGemmaDecisionModel",
    "GemmaClefDecisionModel",
    "SurrogateDecisionModel",
    "MODEL_ALIASES",
    "resolve_model_id",
    "get_backbone_output_dim",
    "compute_calibrated_systemone_loss",
]
