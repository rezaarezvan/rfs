import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from parameters import Params
from dataset import get_data_loader
from model import Generator, Discriminator


def train():
    # Create the data loader
    data_loader = get_data_loader()

    # Create the models
    generator = Generator()
    discriminator = Discriminator()

    # Create the optimizers
    opt_g = optim.Adam(generator.parameters(), lr=Params.learning_rate)
    opt_d = optim.Adam(discriminator.parameters(), lr=Params.learning_rate)

    # Create the loss function
    criterion = nn.BCELoss()

    # Move models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Create a fixed batch of latent vectors that we'll use to visualize
    # the progression of the generator
    fixed_z = torch.randn(64, Params.latent_dim).to(device)

    # Start training
    for epoch in range(Params.epochs):
        for real_images, _ in data_loader:
            real_images = real_images.view(real_images.size(0), -1)
            real_images = real_images.to(device)

            # ============================
            # Train the discriminator
            # ============================

            # Zero the gradients of the discriminator
            discriminator.zero_grad()

            # Compute the discriminator loss on real images
            real_labels = torch.ones(real_images.size(0), 1).to(device)
            real_output = discriminator(real_images)
            d_loss_real = criterion(real_output, real_labels)

            # Generate fake images
            z = torch.randn(real_images.size(0), Params.latent_dim).to(device)
            fake_images = generator(z)

            # Compute the discriminator loss on fake images
            fake_labels = torch.zeros(real_images.size(0), 1).to(device)
            fake_output = discriminator(fake_images)
            d_loss_fake = criterion(fake_output, fake_labels)

            # Compute the total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_d.step()

            # =========================
            # Train the generator
            # =========================

            # Zero the gradients of the generator
            generator.zero_grad()

            # Generate fake images
            z = torch.randn(real_images.size(0), Params.latent_dim).to(device)
            fake_images = generator(z)

            # Compute the generator loss
            output = discriminator(fake_images)
            # We want the generator to fool the discriminator
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            opt_g.step()

        # Save the generated samples
        with torch.no_grad():
            fake_images = generator(fixed_z).view(-1, 28, 28).cpu().numpy()
            plt.figure(figsize=(8, 8))
            for i, img in enumerate(fake_images):
                plt.subplot(8, 8, i + 1)
                plt.imshow(img, cmap='gray')
                plt.axis('off')
            # Save in /imgs/ folder
            plt.savefig(f"imgs/{epoch}.png")
            plt.close()

        print(
            f"Epoch [{epoch}/{Params.epochs}] d_loss: {d_loss.item()} g_loss: {g_loss.item()}")
