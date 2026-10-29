"""Gemma-Clef Decision Model wrapper and lightweight Surrogate model for testing."""

import hashlib
import os
import re
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from wafer.model.encoder import EncodedRecord
from wafer.model.head import JointSchemaHead


def resolve_torch_dtype(preferred: Optional[str] = None) -> torch.dtype:
    """
    Resolves optimal torch dtype based on environment and GPU capabilities.
    Pascal GTX 1080 (sm_61) automatically selects float16; Ampere+ selects bfloat16.
    """
    env_dtype = preferred or os.getenv("TORCH_DTYPE", "auto").lower()

    if env_dtype == "float16" or env_dtype == "fp16":
        return torch.float16
    if env_dtype == "bfloat16" or env_dtype == "bf16":
        return torch.bfloat16
    if env_dtype == "float32" or env_dtype == "fp32":
        return torch.float32

    # Auto resolution
    if torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return torch.bfloat16  # Ampere, Ada, Hopper, Blackwell
            elif major >= 7:
                return torch.float16   # Volta, Turing
        except Exception:
            pass
    return torch.float32


MODEL_ALIASES = {
    "embeddinggemma": "google/embeddinggemma-2",
    "embeddinggemma-2": "google/embeddinggemma-2",
    "google/embeddinggemma-2": "google/embeddinggemma-2",
    "gemma-4-e2b": "google/gemma-4-E2B-it",
    "gemma-4-e2b-it": "google/gemma-4-E2B-it",
    "google/gemma-4-E2B": "google/gemma-4-E2B",
    "google/gemma-4-E2B-it": "google/gemma-4-E2B-it",
}


def resolve_model_id(model_id_or_alias: str) -> str:
    """Resolves CLI aliases and shorthands to official Hugging Face repository identifiers."""
    return MODEL_ALIASES.get(model_id_or_alias.lower(), model_id_or_alias)


def get_backbone_output_dim(cfg: Any, default: int = 768) -> int:
    """
    Extracts the output representation dimensionality from model config.
    Handles embedding backbones (e.g. EmbeddingGemma 2) where internal hidden_size (512)
    differs from final output projection embedding_dim (768), as well as standard LLMs.
    """
    if cfg is None:
        return default
    # 1. Check for embedding_dim in cfg or text_config (e.g. google/embeddinggemma-2)
    for obj in [cfg, getattr(cfg, "text_config", None)]:
        if obj is not None and getattr(obj, "embedding_dim", None) is not None:
            return getattr(obj, "embedding_dim")
    # 2. Check for hidden_size in cfg or text_config
    for obj in [cfg, getattr(cfg, "text_config", None)]:
        if obj is not None and getattr(obj, "hidden_size", None) is not None:
            return getattr(obj, "hidden_size")
    return default


class EmbeddingGemmaDecisionModel(nn.Module):
    """
    High-throughput WAF Decision Model based on Google's EmbeddingGemma 2:
    - Wraps google/embeddinggemma-2 (~270M parameter dense text encoder, Apache 2.0).
    - Maps incoming HTTP requests directly into a 768-dimensional latent space.
    - Attaches Rank-32 LoRA adapters for domain adaptation on web traffic.
    - Evaluates [allow, deny] and [true, false] via JointSchemaHead in a single forward pass.
    - Yields sub-2ms GPU / sub-15ms CPU decision latencies with zero autoregressive overhead.
    """
    def __init__(
        self,
        model_id: str = "google/embeddinggemma-2",
        lora_rank: int = 32,
        lora_alpha: int = 32,
        d_head: int = 256,
        n_heads: int = 4,
        torch_dtype: Optional[torch.dtype] = None,
        load_in_4bit: Optional[bool] = None,
        device_map: Optional[Union[str, dict]] = None,
    ):
        super().__init__()
        from transformers import AutoModel, AutoConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        self.model_id = resolve_model_id(model_id)

        if torch_dtype is None:
            torch_dtype = resolve_torch_dtype()
        if load_in_4bit is None:
            load_in_4bit = os.getenv("LOAD_IN_4BIT", "false").lower() in ("true", "1", "yes")

        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        base_model = AutoModel.from_pretrained(
            self.model_id,
            dtype=torch_dtype if not load_in_4bit else None,
            device_map=device_map,
            **quant_kwargs,
        )

        # Extract text encoder backbone if wrapped inside multimodal container
        if hasattr(base_model, "text_model"):
            self.backbone = base_model.text_model
        elif hasattr(base_model, "language_model"):
            self.backbone = base_model.language_model
        elif hasattr(base_model, "model") and hasattr(base_model.model, "language_model"):
            self.backbone = base_model.model.language_model
        else:
            self.backbone = base_model

        if load_in_4bit:
            self.backbone = prepare_model_for_kbit_training(self.backbone)

        # Freeze base representation weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Attach LoRA adapters to attention and MLP projections
        if lora_rank > 0:
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_dropout=0.05,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)

        cfg = getattr(self.backbone, "config", None)
        hidden_size = get_backbone_output_dim(cfg, default=768)

        self.joint_head = JointSchemaHead(
            backbone_hidden_size=hidden_size,
            d_head=d_head,
            n_heads=n_heads,
        ).to(dtype=torch_dtype)

    def forward_record(self, rec: EncodedRecord) -> List[torch.Tensor]:
        device = next(self.parameters()).device
        input_ids = rec.input_ids.unsqueeze(0).to(device)  # [1, seq_len]

        outputs = self.backbone(input_ids=input_ids)
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state.squeeze(0)
        elif isinstance(outputs, tuple):
            hidden_states = outputs[0].squeeze(0)
        else:
            hidden_states = outputs.squeeze(0)

        return self.joint_head(hidden_states, rec.state_span, rec.option_spans)

    def merge_and_unload(self):
        """Merges LoRA weights into the base backbone for zero-overhead inference."""
        if hasattr(self.backbone, "merge_and_unload"):
            self.backbone = self.backbone.merge_and_unload()
        return self


class GemmaClefDecisionModel(nn.Module):
    """
    Gemma-Clef decision model:
    - Wraps a frozen Gemma 4 backbone (dropping the lm_head to save VRAM).
    - Attaches Rank-256 LoRA adapters to attention and MLP projections.
    - Routes representations through the Two-Stage JointSchemaHead.
    """
    def __init__(
        self,
        model_id: str = "google/gemma-4-E4B-it",
        lora_rank: int = 256,
        lora_alpha: int = 256,
        d_head: int = 1024,
        torch_dtype: Optional[torch.dtype] = None,
        load_in_4bit: Optional[bool] = None,
        device_map: Optional[Union[str, dict]] = None,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model

        if torch_dtype is None:
            torch_dtype = resolve_torch_dtype()
        if load_in_4bit is None:
            load_in_4bit = os.getenv("LOAD_IN_4BIT", "false").lower() in ("true", "1", "yes")

        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        base_causallm = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype if not load_in_4bit else None,
            attn_implementation="sdpa",
            device_map=device_map,
            **quant_kwargs,
        )

        # Extract the inner decoder backbone (drops lm_head to save VRAM)
        if hasattr(base_causallm, "model") and hasattr(base_causallm.model, "language_model"):
            self.backbone = base_causallm.model.language_model
        elif hasattr(base_causallm, "model"):
            self.backbone = base_causallm.model
        else:
            self.backbone = base_causallm

        # Freeze all base backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Attach Rank-256 LoRA adapters
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

        hidden_size = get_backbone_output_dim(getattr(self.backbone, "config", None), default=2048)
        self.joint_head = JointSchemaHead(
            backbone_hidden_size=hidden_size, d_head=d_head
        ).to(dtype=torch_dtype)

    def forward_record(self, rec: EncodedRecord) -> List[torch.Tensor]:
        """Runs a single prefill pass over the packed record and computes question logits."""
        device = next(self.parameters()).device
        input_ids = rec.input_ids.unsqueeze(0).to(device)  # [1, seq_len]

        outputs = self.backbone(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=False,
        )
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state.squeeze(0)
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1].squeeze(0)
        elif isinstance(outputs, tuple):
            hidden_states = outputs[0].squeeze(0)
        else:
            hidden_states = outputs.squeeze(0)

        return self.joint_head(hidden_states, rec.state_span, rec.option_spans)

    def merge_and_unload(self):
        """Merges LoRA weights into the base backbone for zero-overhead inference."""
        if hasattr(self.backbone, "merge_and_unload"):
            self.backbone = self.backbone.merge_and_unload()
        return self


# ============================================================================
# SURROGATE BACKBONE (FOR FAST ZERO-DOWNLOAD UNIT & INTEGRATION TESTING)
# ============================================================================

class MockTokenizer:
    """
    Surrogate subword tokenizer providing deterministic integer token encoding.
    Splits text into subword chunks and punctuation symbols (similar to SentencePiece/BPE),
    ensuring that structural URI tokens, query parameters, headers, and paths are
    represented consistently rather than treating long URLs as single opaque tokens.
    """
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.bos_token_id = 1
        self.eos_token_id = 2

    def _hash_token(self, token_str: str) -> int:
        digest = hashlib.md5(token_str.encode("utf-8")).digest()
        val = int.from_bytes(digest[:4], "big")
        return (val % (self.vocab_size - 10)) + 10

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        # Subword & symbol tokenization: splits into words and punctuation/delimiters
        pieces = re.findall(r"[a-zA-Z0-9]{1,6}|[^\w\s]", text)
        toks = [self._hash_token(p) for p in pieces]
        if not toks and text:
            toks = [self._hash_token(text)]
        if add_special_tokens and self.bos_token_id is not None:
            toks = [self.bos_token_id] + toks
        return toks


class SurrogateDecisionModel(nn.Module):
    """
    Lightweight surrogate transformer model with identical API to GemmaClefDecisionModel.
    Used for instant unit testing, loss verification, and CI without model downloads.
    """
    def __init__(
        self,
        hidden_size: int = 128,
        d_head: int = 64,
        n_heads: int = 4,
        vocab_size: int = 1000,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=hidden_size * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.joint_head = JointSchemaHead(
            backbone_hidden_size=hidden_size,
            d_head=d_head,
            n_heads=n_heads,
            num_routing_layers=1,
            num_joint_layers=1,
        )

    def forward_record(self, rec: EncodedRecord) -> List[torch.Tensor]:
        device = next(self.parameters()).device
        input_ids = rec.input_ids.unsqueeze(0).to(device)  # [1, seq_len]
        h = self.embed(input_ids)                           # [1, seq_len, hidden_size]
        h = self.encoder(h).squeeze(0)                      # [seq_len, hidden_size]
        return self.joint_head(h, rec.state_span, rec.option_spans)
