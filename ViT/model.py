# model.py

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from parameters import *

# Patch Embedding Layer


class PatchEmbedding(nn.Module):
    def __init__(self):
        super(PatchEmbedding, self).__init__()
        self.patch_size = PATCH_SIZE
        self.d_emb = EMBED_DIM

        self.unfold = nn.Unfold(
            kernel_size=self.patch_size, stride=self.patch_size)

        self.projection = nn.Linear(
            self.patch_size*self.patch_size*3, self.d_emb)

    def forward(self, images):
        # images shape: (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
        batch_size = images.shape[0]

        # create patch embeddings
        patches = self.unfold(images)

        # reshape patches
        patches = patches.view(
            batch_size, -1, self.patch_size * self.patch_size * 3)

        # project patches to embeddings
        patches = self.projection(patches)

        return patches


# Vision Transformer


class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()

        # create patch embedding layer
        self.patch_embedding = PatchEmbedding()

        # create transformer encoder
        self.transformer = TransformerEncoder(TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS), num_layers=NUM_LAYERS)

        # classification head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, images):
        # create patch embeddings
        x = self.patch_embedding(images)

        # add cls token
        cls_tokens = self.cls_token.repeat(images.shape[0], 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)

        # pass through transformer
        x = self.transformer(x)

        # take cls token and pass through classifier
        x = x[:, 0]
        x = self.classifier(x)

        return x
