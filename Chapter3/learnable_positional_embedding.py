import torch
from torch import nn


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, vocab_size: int, sequence_size: int, embedding_dim: int):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(p=0.2)
        self.position_embeddings = nn.Embedding(sequence_size, embedding_dim)

    def forward(self, input_token):
        seq_length = input_token.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)

        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_token)
        position_embeddings = self.position_embeddings(position_ids)

        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings