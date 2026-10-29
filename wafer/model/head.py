"""Two-Stage Joint Schema Head (Clef Architecture) for WAF decision making."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Ensure RMSNorm fallback if running on older PyTorch versions
if hasattr(nn, "RMSNorm"):
    RMSNorm = nn.RMSNorm
else:
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            var = torch.mean(x ** 2, dim=-1, keepdim=True)
            normed = x * torch.rsqrt(var + self.eps)
            return normed * self.weight


class EvidenceRoutingLayer(nn.Module):
    """
    Stage 1: Option queries cross-attend over the backbone's HTTP State tokens,
    followed by a residual FFN.
    """
    def __init__(self, d_head: int, n_heads: int, dropout: float = 0.05):
        super().__init__()
        self.norm_q = RMSNorm(d_head)
        self.norm_kv = RMSNorm(d_head)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_head, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = RMSNorm(d_head)
        self.ffn = nn.Sequential(
            nn.Linear(d_head, d_head * 4),
            nn.GELU(),
            nn.Linear(d_head * 4, d_head),
        )

    def forward(self, opt_queries: torch.Tensor, state_kv: torch.Tensor) -> torch.Tensor:
        # opt_queries: [1, total_opts, d_head], state_kv: [1, state_len, d_head]
        q = self.norm_q(opt_queries)
        kv = self.norm_kv(state_kv)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)
        x = opt_queries + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class CrossFieldAttentionLayer(nn.Module):
    """
    Stage 2: Routed option representations self-attend across ALL questions in
    the schema so interdependent fields (e.g., attack_vector & gateway_action)
    maintain joint consistency.
    """
    def __init__(self, d_head: int, n_heads: int, dropout: float = 0.05):
        super().__init__()
        self.norm1 = RMSNorm(d_head)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_head, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = RMSNorm(d_head)
        self.ffn = nn.Sequential(
            nn.Linear(d_head, d_head * 4),
            nn.GELU(),
            nn.Linear(d_head * 4, d_head),
        )

    def forward(self, opt_latents: torch.Tensor) -> torch.Tensor:
        # opt_latents: [1, total_opts, d_head]
        qkv = self.norm1(opt_latents)
        attn_out, _ = self.self_attn(query=qkv, key=qkv, value=qkv, need_weights=False)
        x = opt_latents + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class JointSchemaHead(nn.Module):
    """
    Clef Joint Schema Head:
    1. Attention-weighted pooling of each option token span.
    2. Stage 1: Evidence Routing from HTTP State hidden states.
    3. Stage 2: Cross-Field Joint Self-Attention across all schema options.
    4. Scalar logit projection per option -> list of logits per question.
    """
    def __init__(
        self,
        backbone_hidden_size: int,
        d_head: int = 1024,
        n_heads: int = 8,
        num_routing_layers: int = 2,
        num_joint_layers: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.d_head = d_head
        self.in_proj = nn.Linear(backbone_hidden_size, d_head)
        self.attn_pool_score = nn.Linear(d_head, 1)

        self.routing_layers = nn.ModuleList(
            [EvidenceRoutingLayer(d_head, n_heads, dropout=dropout) for _ in range(num_routing_layers)]
        )
        self.joint_layers = nn.ModuleList(
            [CrossFieldAttentionLayer(d_head, n_heads, dropout=dropout) for _ in range(num_joint_layers)]
        )
        self.final_norm = RMSNorm(d_head)
        self.logit_proj = nn.Linear(d_head, 1)

    def _pool_span(self, h_proj: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Attention-weighted pooling over an option's token span [start, end)."""
        span_tokens = h_proj[start:end]  # [span_len, d_head]
        if span_tokens.size(0) == 0:
            # Fallback for empty spans
            return h_proj.new_zeros(self.d_head)
        if span_tokens.size(0) == 1:
            return span_tokens.squeeze(0)
        weights = F.softmax(self.attn_pool_score(span_tokens), dim=0)  # [span_len, 1]
        return (weights * span_tokens).sum(dim=0)  # [d_head]

    def forward(
        self,
        hidden_states: torch.Tensor,           # [seq_len, backbone_hidden_size]
        state_span: Tuple[int, int],
        option_spans: List[List[Tuple[int, int]]],
    ) -> List[torch.Tensor]:
        """
        Returns a list of 1D logit tensors (one tensor per question, sized [num_opts_q]).
        """
        h = self.in_proj(hidden_states)  # [seq_len, d_head]
        s_start, s_end = state_span
        if s_end <= s_start:
            s_end = s_start + 1
        state_kv = h[s_start:s_end].unsqueeze(0)  # [1, state_len, d_head]

        # Pool every option span across all questions into flat query sequence
        pooled_opts: List[torch.Tensor] = []
        q_option_counts: List[int] = []
        for q_spans in option_spans:
            q_option_counts.append(len(q_spans))
            for (o_start, o_end) in q_spans:
                pooled_opts.append(self._pool_span(h, o_start, o_end))

        if not pooled_opts:
            return []

        opt_latents = torch.stack(pooled_opts, dim=0).unsqueeze(0)  # [1, total_opts, d_head]

        # Stage 1: Route evidence from HTTP state to each option
        for r_layer in self.routing_layers:
            opt_latents = r_layer(opt_latents, state_kv)

        # Stage 2: Joint cross-field self-attention across all options of all questions
        for j_layer in self.joint_layers:
            opt_latents = j_layer(opt_latents)

        # Score options and split back by question
        all_logits = self.logit_proj(self.final_norm(opt_latents)).squeeze(0).squeeze(-1)  # [total_opts]
        return list(torch.split(all_logits, q_option_counts, dim=0))
