import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


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


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()

        self.attention = SelfAttention(dim, heads=heads)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)

        fed_forward = self.feed_forward(x)

        return self.norm2(fed_forward + x)


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=512):
        super().__init__()
        self.dim_model = dim_model

        # Create matrix of [SeqLen, EmbeddingDimension] representing the positional encoding

        pos_enc = torch.zeros(max_len, dim_model)
        for pos in range(max_len):
            for i in range(0, dim_model, 2):
                pos_enc[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i)/dim_model)))
                pos_enc[pos, i + 1] = \
                    math.cos(
                    pos / (10000 ** ((2 * (i + 1))/dim_model)))
        # Include batch dimension
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Make embeddings relatively larger
        x = x * math.sqrt(self.dim_model)
        # Add constant to embedding
        seq_len = x.size(1)
        pos_enc = Variable(self.pos_enc[:, :seq_len], requires_grad=False)

        if x.is_cuda:
            pos_enc.cuda()
        x = x + pos_enc

        return x
