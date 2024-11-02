import time
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms


from tqdm import tqdm
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from rfs import DEVICE
from rfs.models.diffusion import Diffusion, UNet


def get_mnist_loader(batch_size, train=True):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)]
    )

    dataset = MNIST(root="./data", train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(num_epochs=10, batch_size=32):
    dataloader = get_mnist_loader(batch_size=batch_size)

    model = UNet(num_in_channels=1, num_out_channels=1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=28)

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(dataloader)
        pbar.set_description(f"Epoch {epoch}")

        for _, (images, _) in enumerate(pbar):
            images = images.to(DEVICE)

            t = diffusion.sample_timesteps(images.shape[0]).to(DEVICE)

            x_t, noise = diffusion.noise_images(images, t)

            predicted_noise = model(x_t, t)

            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=f"{loss.item():.4f}")

        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}_{time.time()}.pt")

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                sampled_images = diffusion.sample(model, n=16, labels=None)
                torchvision.utils.save_image(
                    sampled_images, f"samples/epoch_{epoch}_{time.time()}.png"
                )


if __name__ == "__main__":
    train(num_epochs=10, batch_size=32)
