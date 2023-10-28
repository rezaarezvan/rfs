import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from parameters import DatasetParams, ModelParams

# Patch Embedding Layer

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=DatasetParams.patch_size, d_emb=ModelParams.embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.d_emb = d_emb
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.projection = nn.Linear(self.patch_size*self.patch_size*3, self.d_emb)

    def forward(self, images):
        batch_size = images.shape[0]
        patches = self.unfold(images)
        patches = patches.view(batch_size, -1, self.patch_size * self.patch_size * 3)
        patches = self.projection(patches)
        return patches

# Vision Transformer

class ViT(nn.Module):
    def __init__(self, d_model=ModelParams.embed_dim, nhead=ModelParams.num_heads, num_layers=ModelParams.num_layers, num_classes=DatasetParams.num_classes):
        super().__init__()
        self.patch_embedding = PatchEmbedding()
        self.transformer = TransformerEncoder(TransformerEncoderLayer(d_model=d_model, nhead=nhead), num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, images):
        x = self.patch_embedding(images)
        cls_tokens = self.cls_token.expand(images.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        x = x[:, 0]
        x = self.classifier(x)
        return x