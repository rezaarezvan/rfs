import torch
import torch.nn as nn
import torch.nn.functional as F

from rfs import DEVICE


class VAE(nn.Module):
    def __init__(self,
                 input_channels=1,
                 latent_dim=4,
                 hidden_dims=None,
                 conditional=False,
                 num_classes=10
                 ):
        super(VAE, self).__init__()

        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.conditional = conditional
        self.num_classes = num_classes

        if hidden_dims is None:
            hidden_dims = [32, 64]
        self.hidden_dims = hidden_dims

        if self.conditional:
            self.input_channels += num_classes

        self.encoder = self.build_encoder(input_channels, self.hidden_dims)
        self.decoder = self.build_decoder(input_channels, self.hidden_dims)

    def build_encoder(self, input_channels, hidden_dims):
        modules = []
        in_channels = input_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3,
                              stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        encoder = nn.Sequential(*modules)

        self._compute_latent_shape(encoder, input_channels)
        self.fc_mu = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_dim, self.latent_dim)

        return encoder

    def _compute_latent_shape(self, encoder, input_channels):
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 28, 28)
            encoder_output = encoder(dummy_input)
            self.last_shape = encoder_output.shape[1:]
            self.flatten_dim = encoder_output.numel()

    def build_decoder(self, input_channels, hidden_dims):
        hidden_dims = hidden_dims[::-1]
        modules = []
        in_channels = hidden_dims[0]

        input_dim = self.latent_dim + \
            self.num_classes if self.conditional else self.latent_dim

        self.decoder_input = nn.Linear(input_dim, self.flatten_dim)

        for i in range(len(hidden_dims) - 1):
            out_channels = hidden_dims[i + 1]
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels,
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU())
            )
            in_channels = out_channels

        decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, input_channels,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        return decoder

    def encode(self, input, labels=None):
        if self.conditional and labels is not None:
            y = F.one_hot(labels, num_classes=self.num_classes).float()
            y = y.view(y.size(0), self.num_classes, 1, 1)
            y = y.expand(-1, -1, input.size(2), input.size(3))
            input = torch.cat([input, y], dim=1)

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels=None):
        if self.conditional and labels is not None:
            y = F.one_hot(labels, num_classes=self.num_classes).float()
            z = torch.cat([z, y], dim=1)

        result = self.decoder_input(z)
        result = result.view(-1, *self.last_shape)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input, labels=None):
        mu, log_var = self.encode(input, labels)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z, labels)
        return reconstructed, mu, log_var

    def sample(self, num_samples, labels=None):
        z = torch.randn(num_samples, self.latent_dim).to(DEVICE)
        samples = self.decode(z, labels)
        return samples
