import torch
import torch.nn as nn

from rfs.utils.dit_utils import extract_patches, reconstruct_image


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000, device=device)).float() / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiTBlock(nn.Module):
    """
    DiT Block from the paper, implementing the adaptive Layer Norm (adaLN) mechanism
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.attn.in_proj_weight, std=0.02)
        nn.init.normal_(self.attn.out_proj.weight, std=0.02)
        nn.init.zeros_(self.attn.in_proj_bias)
        nn.init.zeros_(self.attn.out_proj.bias)

        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].bias)

        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, t):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(t).chunk(6, dim=1)
        )

        norm_x = self.norm1(x)
        norm_x = norm_x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(norm_x, norm_x, norm_x)[0]

        norm_x = self.norm2(x)
        norm_x = norm_x * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(norm_x)

        return x


class DiT(nn.Module):
    """
    DiT-B/2 architecture from the paper "Scalable Diffusion Models with Transformers"
    https://arxiv.org/abs/2212.09748
    """

    def __init__(
        self,
        image_size,
        channels_in,
        patch_size=2,
        hidden_size=768,
        num_heads=12,
        num_layers=12,
        mlp_ratio=4.0,
        num_classes=None,
    ):
        super().__init__()
        self.image_size = image_size
        self.channels_in = channels_in
        self.patch_size = patch_size
        self.num_classes = num_classes

        self.x_embedder = nn.Sequential(
            nn.Linear(channels_in * patch_size * patch_size, hidden_size),
            nn.GELU(),
        )

        self.time_embedder = nn.Sequential(
            SinusoidalPosEmb(hidden_size),
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

        if num_classes is not None:
            self.class_embedder = nn.Embedding(num_classes, hidden_size)

        self.num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_final = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(
            hidden_size, channels_in * patch_size * patch_size
        )

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.x_embedder[0].weight, std=0.02)
        nn.init.zeros_(self.x_embedder[0].bias)

        nn.init.normal_(self.pos_embed, std=0.02)

        nn.init.normal_(self.output_projection.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias)

    def unpatchify(self, x):
        """Convert patched feature back to image."""
        return reconstruct_image(
            x,
            (x.shape[0], self.channels_in, self.image_size, self.image_size),
            self.patch_size,
        )

    def forward(self, x, t, y=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels when class conditioning is used
        """
        x = extract_patches(x, self.patch_size)
        x = self.x_embedder(x)

        x = x + self.pos_embed

        t = self.time_embedder(t)

        if y is not None and self.num_classes is not None:
            t = t + self.class_embedder(y)

        for block in self.blocks:
            x = block(x, t)

        x = self.norm_final(x)
        x = self.output_projection(x)

        return self.unpatchify(x)
