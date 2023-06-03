import torch
import torch.nn.functional as F
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads

        # These compute the keys, queries and values for all
        # heads (as a single concatenated vector)

        self.to_keys = nn.Linear(dim, dim * heads, bias=False)
        self.to_queries = nn.Linear(dim, dim * heads, bias=False)
        self.to_values = nn.Linear(dim, dim * heads, bias=False)

        # This unifies the outputs of the different heads into
        # a single k-vector

        self.unify_heads = nn.Linear(heads * dim, dim)

    def forward(self, x):
        batch_size, sequence_length, dim = x.size()
        heads = self.heads

        queries = self.to_queries(x).view(
            batch_size, sequence_length, heads, dim)
        keys = self.to_keys(x).view(
            batch_size, sequence_length, heads, dim)
        values = self.to_values(x).view(
            batch_size, sequence_length, heads, dim)

        queries = queries.transpose(1, 2).contiguous().view(
            batch_size * heads, sequence_length, dim)
        keys = keys.transpose(1, 2).contiguous().view(
            batch_size * heads, sequence_length, dim)
        values = values.transpose(1, 2).contiguous().view(
            batch_size * heads, sequence_length, dim)

        queries = queries / (dim ** (1 / 4))
        keys = keys / (dim ** (1 / 4))

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        # - dot has size (batch_size * heads, sequence_length, sequence_length) containing raw weights
        dot = F.softmax(dot, dim=2)

        # - dot now contains row-wise normalized weights

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(
            batch_size, heads, sequence_length, dim)

        # swap heads and sequence length back, unify heads
        out = out.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, heads * dim)

        return self.unify_heads(out)
