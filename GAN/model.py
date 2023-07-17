import torch
from torch import nn
from parameters import Params


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            # input is a noise vector of size latent_dim
            nn.Linear(Params.latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, Params.image_dim),
            nn.Tanh()  # output is an image
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            # input is an image
            nn.Linear(Params.image_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()  # output is a probability that the image is real
        )

    def forward(self, input):
        return self.main(input)
