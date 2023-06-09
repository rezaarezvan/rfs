from torch import nn
from . import layers


class Transformer(nn.Module):
    def __init__(self, dim, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_enc = layers.PositionalEncoding(dim, seq_length)

        # The sequence of transformer blocks that does all the
        # heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(layers.TransformerBlock(dim=dim, heads=heads))

        self.tblocks = nn.Sequential(*tblocks)

        # Maps the final output sequence to class logits
        self.toprobs = nn.Linear(dim, num_classes)

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the
                 classes (where c is the nr. of classes).
        """
        print(x.shape)

        # Generate token embeddings
        tokens = self.token_emb(x)
        print(x.shape)

        # Apply positional encoding and feed tokens into the transformer blocks.
        x = self.pos_enc(tokens)
        print(x.shape)
        x = self.tblocks(x)
        print(x.shape)

        # Average pool over the time dimension and project to class probabilities
        x = self.toprobs(x).mean(dim=1)
        print(x.shape)

        return nn.functional.log_softmax(x, dim=1)
