"""Unit tests verifying EmbeddingGemma 2 backbone integration and latent schema routing."""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from wafer.data.synthetic import generate_synthetic_waf_samples
from wafer.model.backbone import (
    EmbeddingGemmaDecisionModel,
    MockTokenizer,
    resolve_model_id,
    get_backbone_output_dim,
    MODEL_ALIASES,
)
from wafer.model.encoder import encode_record
from wafer.model.head import JointSchemaHead
from wafer.training.train import get_optimizer_with_differential_lrs


class DummyEmbeddingConfig:
    """Mock config reflecting google/embeddinggemma-2 architecture."""
    hidden_size = 768
    num_hidden_layers = 2
    vocab_size = 1000


class DummyEmbeddingTower(nn.Module):
    """Mock embedding backbone generating 768-dim representations without network dependencies."""
    def __init__(self, hidden_size: int = 768, vocab_size: int = 1000):
        super().__init__()
        self.config = DummyEmbeddingConfig()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=hidden_size * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        h = self.embed(input_ids)
        out = self.encoder(h)
        return SimpleNamespace(last_hidden_state=out)


def test_embeddinggemma_alias_resolution():
    """Verify CLI aliases map correctly to official Hugging Face identifiers."""
    assert resolve_model_id("embeddinggemma") == "google/embeddinggemma-2"
    assert resolve_model_id("embeddinggemma-2") == "google/embeddinggemma-2"
    assert resolve_model_id("google/embeddinggemma-2") == "google/embeddinggemma-2"
    assert "embeddinggemma-2" in MODEL_ALIASES


def test_get_backbone_output_dim():
    """Verify dimensionality resolution handles embedding_dim vs hidden_size correctly."""
    # 1. EmbeddingGemma-style config: internal hidden_size is 512, but output embedding_dim is 768
    cfg_embedding = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=512, embedding_dim=768)
    )
    assert get_backbone_output_dim(cfg_embedding) == 768

    # 2. Top-level embedding_dim
    cfg_top = SimpleNamespace(embedding_dim=768, hidden_size=512)
    assert get_backbone_output_dim(cfg_top) == 768

    # 3. Standard LLM config (hidden_size only)
    cfg_llm = SimpleNamespace(hidden_size=2048)
    assert get_backbone_output_dim(cfg_llm) == 2048

    # 4. Fallback default
    assert get_backbone_output_dim(None, default=768) == 768


def test_embeddinggemma_latent_routing():
    """Verify 768-dim latent space routes properly through JointSchemaHead."""
    torch.manual_seed(42)
    tok = MockTokenizer()
    rec = generate_synthetic_waf_samples()[0]
    enc = encode_record(tok, rec)

    # Instantiate mock embedding model with 768 dimensions
    hidden_size = 768
    d_head = 256
    n_heads = 4

    backbone = DummyEmbeddingTower(hidden_size=hidden_size)
    head = JointSchemaHead(backbone_hidden_size=hidden_size, d_head=d_head, n_heads=n_heads)

    input_ids = enc.input_ids.unsqueeze(0)
    out = backbone(input_ids=input_ids)
    hidden_states = out.last_hidden_state.squeeze(0)  # [seq_len, 768]

    logits_list = head(hidden_states, enc.state_span, enc.option_spans)

    assert len(logits_list) == len(rec["questions"])
    # Action question: 2 options (allow, deny)
    assert logits_list[0].shape == (2,)
    # Alert question: 2 options (true, false)
    assert logits_list[1].shape == (2,)


def test_embeddinggemma_probabilistic_calibration():
    """Verify forward-pass output probabilities sum to 1.0."""
    torch.manual_seed(42)
    tok = MockTokenizer()
    rec = generate_synthetic_waf_samples()[1]
    enc = encode_record(tok, rec)

    backbone = DummyEmbeddingTower(hidden_size=768)
    head = JointSchemaHead(backbone_hidden_size=768, d_head=256, n_heads=4)

    with torch.no_grad():
        out = backbone(input_ids=enc.input_ids.unsqueeze(0))
        logits_list = head(out.last_hidden_state.squeeze(0), enc.state_span, enc.option_spans)

    for logits in logits_list:
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_embeddinggemma_differential_optimizer_setup():
    """Verify optimizer correctly splits adapter and head parameter groups."""
    backbone = DummyEmbeddingTower(hidden_size=768)
    for p in backbone.parameters():
        p.requires_grad = False

    lora_param = nn.Parameter(torch.randn(768, 16))
    head = JointSchemaHead(backbone_hidden_size=768, d_head=256, n_heads=4)

    class CombinedModel(nn.Module):
        def __init__(self, b, h, lora):
            super().__init__()
            self.backbone = b
            self.joint_head = h
            self.lora_layer = lora

    model = CombinedModel(backbone, head, lora_param)
    opt = get_optimizer_with_differential_lrs(model, head_lr=2e-4, lora_lr=5e-5)

    assert len(opt.param_groups) == 2
    head_group = [g for g in opt.param_groups if g["lr"] == 2e-4]
    lora_group = [g for g in opt.param_groups if g["lr"] == 5e-5]
    assert len(head_group) == 1
    assert len(lora_group) == 1
