import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    n_layers: int = 13
    d_model: int = 640
    n_heads: int = 10
    ffn_dim: int = 2560
    max_seq_len: int = 512
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normalized = (x_fp32 * torch.rsqrt(variance + self.eps)).to(x.dtype)
        return normalized * self.weight


def apply_rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    head_dim = x.shape[-1]
    frequencies = torch.arange(
        0, head_dim, 2, device=x.device, dtype=torch.float32
    )
    inv_freq = 1.0 / (theta ** (frequencies / head_dim))
    positions = torch.arange(
        x.shape[-2], device=x.device, dtype=torch.float32
    )
    angles = torch.outer(positions, inv_freq)
    cos = angles.cos().to(x.dtype)[None, None, :, :]
    sin = angles.sin().to(x.dtype)[None, None, :, :]

    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos), dim=-1
    )
    return rotated.flatten(-2)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.rope_theta = config.rope_theta
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        shape = (batch_size, seq_len, self.n_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q = apply_rope(q, self.rope_theta)
        k = apply_rope(k, self.rope_theta)

        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=0.0
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        return self.o_proj(attended)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class TinyLM(nn.Module):
    def __init__(
        self, config: ModelConfig, activation_checkpointing: bool = True
    ):
        super().__init__()
        dimensions = (
            config.vocab_size,
            config.n_layers,
            config.d_model,
            config.n_heads,
            config.ffn_dim,
            config.max_seq_len,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("Model dimensions and sequence length must be positive")
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if (config.d_model // config.n_heads) % 2 != 0:
            raise ValueError("Attention head dimension must be even")
        if config.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

        self.config = config
        self.activation_checkpointing = activation_checkpointing
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            Block(config) for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must contain int64 token indices")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("Input sequence exceeds max_seq_len")

        x = self.token_embedding(input_ids)
        for block in self.blocks:
            if self.training and self.activation_checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.final_norm(x)
        return F.linear(x, self.token_embedding.weight)

