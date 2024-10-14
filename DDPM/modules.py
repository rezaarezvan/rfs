import torch
import torch.nn as nn
import torch.nn.functional as F


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
            ema_param.data = self.beta * ema_param.data + \
                (1 - self.beta) * param.data

        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class DoubleConv(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.mid_channels = self.num_out_channels if not mid_channels else mid_channels
        self.residual = residual

        self.double_conv = nn.Sequential(
            nn.Conv2d(self.num_in_channels, self.mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, self.mid_channels),
            nn.GELU(),
            nn.Conv2d(self.mid_channels, self.num_out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, self.num_out_channels),
        )

    def forward(self, x):
        return F.gelu(x + self.double_conv(x)) if self.residual else self.double_conv(x)


class Down(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, emb_dim=256):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.emb_dim = emb_dim

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(self.num_in_channels,
                       self.num_in_channels, residual=True),
            DoubleConv(self.num_in_channels, self.num_out_channels)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                self.emb_dim,
                self.num_out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, emb_dim=256):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.emb_dim = emb_dim

        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True)

        self.conv = nn.Sequential(
            DoubleConv(self.num_in_channels,
                       self.num_in_channels, residual=True),
            DoubleConv(self.num_in_channels, self.num_out_channels,
                       self.num_in_channels // 2)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                self.emb_dim,
                self.num_out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class SelfAttention(nn.Module):
    def __init__(self, num_channels, size):
        super(SelfAttention, self).__init__()
        self.num_channels = num_channels
        self.size = size

        self.mha = nn.MultiheadAttention(
            self.num_channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([self.num_channels])
        self.ff_self = nn.Sequential(
            self.ln,
            nn.Linear(self.num_channels, self.num_channels),
            nn.GELU(),
            nn.Linear(self.num_channels, self.num_channels)
        )

    def forward(self, x):
        x = x.view(-1, self.num_channels, self.size*self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.num_channels, self.size, self.size)


class UNet(nn.Module):
    def __init__(self, num_in_channels=3, num_out_channels=3, time_dim=256):
        super().__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.time_dim = time_dim

        # Encoder
        self.inc = DoubleConv(self.num_in_channels, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        # Bottleneck
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # Decoder
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, self.num_out_channels, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / \
            (10_000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))

        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)

        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

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


class UNetConditional(nn.Module):
    def __init__(self, num_in_channels=3, num_out_channels=3, time_dim=256, num_classes=None):
        super().__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.time_dim = time_dim

        # Encoder
        self.inc = DoubleConv(self.num_in_channels, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        # Bottleneck
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # Decoder
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, self.num_out_channels, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(self.num_classes, self.time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / \
            (10_000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))

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
