"""Gemma-Clef model components and loss functions."""

from wafer.model.encoder import EncodedRecord, encode_record
from wafer.model.head import EvidenceRoutingLayer, CrossFieldAttentionLayer, JointSchemaHead
from wafer.model.backbone import GemmaClefDecisionModel, SurrogateDecisionModel
from wafer.model.loss import compute_calibrated_systemone_loss

__all__ = [
    "EncodedRecord",
    "encode_record",
    "EvidenceRoutingLayer",
    "CrossFieldAttentionLayer",
    "JointSchemaHead",
    "GemmaClefDecisionModel",
    "SurrogateDecisionModel",
    "compute_calibrated_systemone_loss",
]
