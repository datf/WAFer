"""Unit tests for JointSchemaHead and Clef routing layers."""

import pytest
import torch

from wafer.model.head import (
    EvidenceRoutingLayer,
    CrossFieldAttentionLayer,
    JointSchemaHead,
)


def test_evidence_routing_layer():
    d_head = 64
    n_heads = 4
    layer = EvidenceRoutingLayer(d_head=d_head, n_heads=n_heads)

    # opt_queries: [1, 5, 64], state_kv: [1, 20, 64]
    queries = torch.randn(1, 5, d_head)
    state_kv = torch.randn(1, 20, d_head)

    out = layer(queries, state_kv)
    assert out.shape == queries.shape
    assert not torch.isnan(out).any()


def test_cross_field_attention_layer():
    d_head = 64
    n_heads = 4
    layer = CrossFieldAttentionLayer(d_head=d_head, n_heads=n_heads)

    opt_latents = torch.randn(1, 7, d_head)
    out = layer(opt_latents)
    assert out.shape == opt_latents.shape
    assert not torch.isnan(out).any()


def test_joint_schema_head_forward_and_backward():
    backbone_dim = 128
    d_head = 64
    n_heads = 4
    seq_len = 50

    head = JointSchemaHead(
        backbone_hidden_size=backbone_dim,
        d_head=d_head,
        n_heads=n_heads,
        num_routing_layers=1,
        num_joint_layers=1,
    )

    hidden_states = torch.randn(seq_len, backbone_dim, requires_grad=True)
    state_span = (5, 20)
    # Question 1: 2 options; Question 2: 3 options
    option_spans = [
        [(22, 25), (26, 30)],
        [(32, 35), (36, 40), (41, 45)],
    ]

    logits_list = head(hidden_states, state_span, option_spans)

    # Verify output list structure
    assert isinstance(logits_list, list)
    assert len(logits_list) == 2
    assert logits_list[0].shape == (2,)
    assert logits_list[1].shape == (3,)

    # Verify gradient flow
    total = logits_list[0].sum() + logits_list[1].sum()
    total.backward()
    assert hidden_states.grad is not None
    assert not torch.isnan(hidden_states.grad).any()
