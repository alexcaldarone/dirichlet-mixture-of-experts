import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .router import DirichletRouter


@dataclass
class DirMoEConfig:
    # dimensions
    vocab_size: int = 32000
    d_model: int = 512
    num_layers: int = 12
    num_heads: int = 8
    num_kv_heads: int = 4
    d_ffn: int = 1408
    max_seq_len: int = 2048
    # MoE config
    num_experts: int = 8
    k: int = 1
    hidden_dim_router: int = 256
    # Router hyperparameters (paper)
    lambda_p: float = 0.5
    lambda_q: float = 20.0
    tau_z: float = 2.0
    alpha_hi_prior: float = 1.985
    alpha_lo_prior: float = 0.005
    # Loss hyperparameters
    beta_theta: float = 0.01
    lambda_sparsity: float = 0.01
    sigma2: float = 1.0
    # Other
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(self, seq_len: int):
        return self.cos_cache[:seq_len], self.sin_cache[:seq_len]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, num_heads, T, head_dim); cos/sin: (T, head_dim//2)"""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, d//2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: DirMoEConfig):
        super().__init__()
        assert config.d_model % config.num_heads == 0
        assert config.num_heads % config.num_kv_heads == 0
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.kv_groups = config.num_heads // config.num_kv_heads
        self.head_dim = config.d_model // config.num_heads
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.d_model, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Expand KV heads to match query head count
        k = k.repeat_interleave(self.kv_groups, dim=1)
        v = v.repeat_interleave(self.kv_groups, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, D))


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.up_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Decoder(nn.Module):
    def __init__(self, num_experts: int, hidden_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_experts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)


class MoELayer(nn.Module):
    def __init__(self, config: DirMoEConfig):
        super().__init__()
        self.router = DirichletRouter(
            input_dim=config.d_model,
            hidden_dim=config.hidden_dim_router,
            num_experts=config.num_experts,
            lambda_p=config.lambda_p,
            lambda_q=config.lambda_q,
            tau_z=config.tau_z,
            alpha_hi_prior=config.alpha_hi_prior,
            alpha_lo_prior=config.alpha_lo_prior,
        )
        self.experts = nn.ModuleList([
            ExpertFFN(config.d_model, config.d_ffn) for _ in range(config.num_experts)
        ])
        self.decoder = Decoder(config.num_experts, config.hidden_dim_router, config.d_model)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            y:   (B, T, d_model) — weighted sum of expert outputs
            aux: dict with router outputs needed for DirMoELoss
        """
        B, T, D = x.shape
        x_flat = x.view(B * T, D)  # (N, D)

        r, z, theta, alpha_p, alpha_q = self.router(x_flat)

        # Naive: run every expert on every token, then mix. (N, E, D) → (N, D)
        expert_out = torch.stack([e(x_flat) for e in self.experts], dim=1)
        y = (r.unsqueeze(-1) * expert_out).sum(dim=1)  # (N, D)

        x_recon = self.decoder(r)  # (N, D)

        aux = {
            "z": z,
            "alpha_p": alpha_p,
            "alpha_q": alpha_q,
            "x_orig": x_flat.detach(),
            "x_recon": x_recon,
        }
        return y.view(B, T, D), aux


class TransformerBlock(nn.Module):
    def __init__(self, config: DirMoEConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.moe = MoELayer(config)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.attn_norm(x))
        h, aux = self.moe(self.ffn_norm(x))
        x = x + h
        return x, aux


class DirMoE(nn.Module):
    def __init__(self, config: DirMoEConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        """
        Args:
            input_ids: (B, T) token indices
        Returns:
            logits:   (B, T, vocab_size)
            aux_list: list of per-layer aux dicts for DirMoELoss
        """
        x = self.embed(input_ids)
        aux_list = []
        for block in self.blocks:
            x, aux = block(x)
            aux_list.append(aux)
        logits = self.lm_head(self.norm(x))
        return logits, aux_list


# Loss functions
def _dirichlet_kl(alpha_q: torch.Tensor, alpha_p: torch.Tensor) -> torch.Tensor:
    sum_q = alpha_q.sum(-1, keepdim=True)
    sum_p = alpha_p.sum(-1, keepdim=True)
    kl = (
        torch.lgamma(sum_q) - torch.lgamma(alpha_q).sum(-1, keepdim=True)
        - torch.lgamma(sum_p) + torch.lgamma(alpha_p).sum(-1, keepdim=True)
        + ((alpha_q - alpha_p) * (torch.digamma(alpha_q) - torch.digamma(sum_q))).sum(-1, keepdim=True)
    )
    return kl.squeeze(-1)  # (N,)


class DirMoELoss(nn.Module):
    """
    Total loss: L_LM + (1/L) Σ_layers [L_recon + β_θ * KL_Dir + R_sparsity]

    Args:
        k:               target number of active experts per token
        beta_theta:      weight for Dirichlet KL term (paper default 0.01)
        lambda_sparsity: weight for sparsity penalty (paper default 0.01)
        sigma2:          reconstruction variance (paper default 1.0)
    """

    def __init__(self, k: int, beta_theta: float = 0.01,
                 lambda_sparsity: float = 0.01, sigma2: float = 1.0):
        super().__init__()
        self.k = k
        self.beta_theta = beta_theta
        self.lambda_sparsity = lambda_sparsity
        self.sigma2 = sigma2

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, aux_list: list) -> torch.Tensor:
        """
        Args:
            logits:   (B, T, vocab_size)
            targets:  (B, T) token indices (typically input_ids shifted by 1)
            aux_list: list of per-layer dicts from DirMoE.forward()
        Returns:
            scalar total loss
        """
        B, T, V = logits.shape
        L_lm = F.cross_entropy(logits.view(-1, V), targets.view(-1))

        L_dirmoe = logits.new_zeros(1)
        for aux in aux_list:
            z = aux["z"]            # (N, E)
            alpha_p = aux["alpha_p"]
            alpha_q = aux["alpha_q"]
            x_orig = aux["x_orig"]  # (N, D) — detached
            x_recon = aux["x_recon"]

            L_recon = (x_orig - x_recon).pow(2).sum(-1).mean() / (2.0 * self.sigma2)
            L_kl = _dirichlet_kl(alpha_q, alpha_p).mean()
            R_sparsity = self.lambda_sparsity * (z.sum(-1) - self.k).pow(2).mean()

            L_dirmoe = L_dirmoe + L_recon + self.beta_theta * L_kl + R_sparsity

        return L_lm + L_dirmoe / len(aux_list)
