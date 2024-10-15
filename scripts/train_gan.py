import os
import torch

from torch import nn, optim
from torchvision.utils import save_image
from rfs.models.gan import Generator, Discriminator
from rfs.data.dataloaders import get_mnist_loader


def main():
    # Hyperparameters
    epochs = 50
    batch_size = 128
    learning_rate = 0.0002
    latent_dim = 100
    image_dim = 28 * 28

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(latent_dim, image_dim).to(device)
    discriminator = Discriminator(image_dim).to(device)

    opt_g = optim.Adam(generator.parameters(), lr=learning_rate)
    opt_d = optim.Adam(discriminator.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    train_loader = get_mnist_loader(batch_size=batch_size, train=True)
    fixed_noise = torch.randn(64, latent_dim).to(device)

    os.makedirs('imgs', exist_ok=True)

    for epoch in range(epochs):
        for real, _ in train_loader:
            real = real.view(-1, image_dim).to(device)
            batch_size = real.size(0)

            ### Train Discriminator ###
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake = generator(noise)

            disc_real = discriminator(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))

            disc_fake = discriminator(fake.detach()).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            lossD = (lossD_real + lossD_fake) / 2
            discriminator.zero_grad()
            lossD.backward()
            opt_d.step()

            ### Train Generator ###
            output = discriminator(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            generator.zero_grad()
            lossG.backward()
            opt_g.step()

        print(f"Epoch [{epoch+1}/{epochs}] Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")

        with torch.no_grad():
            fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
            save_image(fake, os.path.join('imgs', f"{epoch+1}.png"))


if __name__ == '__main__':
    main()
