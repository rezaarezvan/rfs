import torch
import torch.nn as nn

from tqdm import tqdm
from rfs import DEVICE
from rfs.utils.dit_utils import extract_patches, reconstruct_image


class Diffusion:
    def __init__(
        self, num_noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64
    ):
        self.num_noise_steps = num_noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = DEVICE

        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.num_noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.num_noise_steps, size=(n,))

    def sample(self, model, n, labels=None, cfg_scale=3):
        model.eval()
        with torch.no_grad():
            channels = model.inc.double_conv[0].in_channels
            x = torch.randn((n, channels, self.img_size, self.img_size)).to(self.device)

            for i in tqdm(reversed(range(1, self.num_noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)

                predicted_noise = model(x, t)
                if labels is not None and cfg_scale > 0:
                    unconditional = model(x, t, None)
                    predicted_noise = torch.lerp(
                        unconditional, predicted_noise, cfg_scale
                    )

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                x = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )

            x = (x.clamp(-1, 1) + 1) / 2

            return x


class EMA:
    def __init__(self, beta=0.99):
        self.beta = beta
        self.step = 0

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return

        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data = self.beta * ema_param.data + (1 - self.beta) * param.data

        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return nn.GELU()(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t_emb):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t_emb)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1]
        )
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t_emb):
        x = self.up(x)
        if x.shape != skip_x.shape:
            x = torch.nn.functional.interpolate(x, size=skip_x.shape[2:])
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t_emb)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1]
        )
        return x + emb


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).transpose(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.transpose(1, 2).view(b, c, h, w)


class UNet(nn.Module):
    def __init__(self, num_in_channels=1, num_out_channels=1, time_dim=256):
        super().__init__()
        self.device = DEVICE
        self.time_dim = time_dim

        self.inc = DoubleConv(num_in_channels, 64)

        # Encoder
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)

        # Bottleneck
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # Decoder
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)

        # Output projection
        self.outc = nn.Conv2d(64, num_out_channels, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        # Bottleneck
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # Decoder path
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        return self.outc(x)


class UNetConditional(nn.Module):
    def __init__(
        self, num_in_channels=3, num_out_channels=3, time_dim=256, num_classes=None
    ):
        super().__init__()
        self.device = DEVICE

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.time_dim = time_dim

        # Encoder
        self.inc = DoubleConv(self.num_in_channels, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)

        # Bottleneck
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # Decoder
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, self.num_out_channels, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(self.num_classes, self.time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10_000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )

        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)

        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        output = self.outc(x)

        return output


class ConditionalNorm2d(nn.Module):
    def __init__(self, hidden_size, num_features):
        super(ConditionalNorm2d, self).__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.fcw = nn.Linear(num_features, hidden_size)
        self.fcb = nn.Linear(num_features, hidden_size)

    def forward(self, x, features):
        bs, s, l = x.shape

        out = self.norm(x)
        w = self.fcw(features).reshape(bs, 1, -1)
        b = self.fcb(features).reshape(bs, 1, -1)

        return w * out + b


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


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, num_features=128):
        super(TransformerBlock, self).__init__()
        self.norm = nn.LayerNorm(hidden_size)

        self.multihead_attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True, dropout=0.0
        )

        self.con_norm = ConditionalNorm2d(hidden_size, num_features)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LayerNorm(hidden_size * 4),
            nn.ELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x, features):
        norm_x = self.norm(x)
        x = self.multihead_attn(norm_x, norm_x, norm_x)[0] + x
        norm_x = self.con_norm(x, features)
        x = self.mlp(norm_x) + x

        return x


class DiT(nn.Module):
    def __init__(
        self,
        image_size,
        channels_in,
        patch_size=16,
        hidden_size=128,
        num_features=128,
        num_layers=3,
        num_heads=4,
    ):
        super(DiT, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(num_features),
            nn.Linear(num_features, 2 * num_features),
            nn.GELU(),
            nn.Linear(2 * num_features, num_features),
            nn.GELU(),
        )

        self.patch_size = patch_size
        self.fc_in = nn.Linear(channels_in * patch_size * patch_size, hidden_size)

        seq_length = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_size).normal_(std=0.02)
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(hidden_size, channels_in * patch_size * patch_size)

    def forward(self, image_in, index):
        index_features = self.time_mlp(index)

        patch_seq = extract_patches(image_in, patch_size=self.patch_size)
        patch_emb = self.fc_in(patch_seq)

        embs = patch_emb + self.pos_embedding

        for block in self.blocks:
            embs = block(embs, index_features)

        image_out = self.fc_out(embs)

        return reconstruct_image(image_out, image_in.shape, patch_size=self.patch_size)
