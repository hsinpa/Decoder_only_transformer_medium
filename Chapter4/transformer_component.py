from dataclasses import dataclass

import torch
import torch.nn as nn

from Chapter1.attention_component import MultiHeadAttention
from Chapter3.learnable_positional_embedding import LearnablePositionalEncoding


@dataclass
class TransformerConfig:
    embed_dim: int
    window_size: int
    vocab_size: int

    attention_head_size: int
    attention_layer_size: int
    hidden_dropout_prob: float

    inference_mode: bool
    device: torch.device


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.embeddings = LearnablePositionalEncoding(vocab_size=config.vocab_size,
                                                                    sequence_size=config.window_size,
                                                                    embedding_dim=config.embed_dim, )

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.attention_layer_size)])
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.output_layer = nn.Linear(config.embed_dim, config.vocab_size, bias=True)

    def forward(self, x):
        x = self.embeddings(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        x = self.output_layer(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm_2 = nn.LayerNorm(config.embed_dim)

        self.attention = MultiHeadAttention(window_size=config.window_size, embed_dim=config.embed_dim,
                                            num_heads=config.attention_head_size, mask=config.inference_mode)

        self.feed_forward = FeedForward(hidden_size=config.embed_dim, intermediate_size=config.embed_dim * 4,
                                        hidden_dropout_prob=config.hidden_dropout_prob)

    def forward(self, x):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_dropout_prob: float):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, intermediate_size)
        self.linear_2 = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm_2 = nn.LayerNorm(config.embed_dim)

        self.attention = MultiHeadAttention(window_size=config.window_size, embed_dim=config.embed_dim,
                                            num_heads=config.attention_head_size, mask=config.inference_mode)

        self.feed_forward = FeedForward(hidden_size=config.embed_dim, intermediate_size=config.embed_dim * 4,
                                        hidden_dropout_prob=config.hidden_dropout_prob)

    def forward(self, x):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x