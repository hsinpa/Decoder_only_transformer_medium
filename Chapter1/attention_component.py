from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                 mask: torch.Tensor = None):
    dim_k = key.size(-1)

    scores = torch.bmm(query, key.transpose(2, 1)) / sqrt(dim_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)

    attn_outputs = torch.bmm(weights, value)
    return attn_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, window_size: int, embed_dim: int, num_heads: int, mask: bool):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(window_size, embed_dim, head_dim, mask) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x


class AttentionHead(nn.Module):
    def __init__(self, window_size: int, embed_dim: int, head_dim: int, mask: bool = False):
        super().__init__()
        self.is_masking = mask
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

        attention_mask = torch.tril(torch.ones(window_size, window_size)).unsqueeze(0)
        self.register_buffer("mask", attention_mask)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))

        return attn_outputs
