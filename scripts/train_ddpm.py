import time
import wandb
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms


from tqdm import tqdm
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from rfs import DEVICE
from rfs.models.ddpm import DDPM, UNet


def get_mnist_loader(batch_size, train=True):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)]
    )

    dataset = MNIST(root="./data", train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(num_epochs=10, batch_size=32):
    wandb.init(
        project="diffusion_experiment",
        name="dppm_mnist_run1",
        config={
            "architecture": "UNet",
            "dataset": "MNIST",
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": 3e-4,
            "device": str(DEVICE),
            "img_size": 28,
        },
    )

    dataloader = get_mnist_loader(batch_size=batch_size)

    model = UNet(num_in_channels=1, num_out_channels=1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    mse = nn.MSELoss()
    diffusion = DDPM(img_size=28)

    wandb.watch(model, log_freq=100)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader)
        pbar.set_description(f"Epoch {epoch}")

        global_step = 0
        for _, (images, _) in enumerate(pbar):
            images = images.to(DEVICE)

            t = diffusion.sample_timesteps(images.shape[0]).to(DEVICE)

            x_t, noise = diffusion.noise_images(images, t)

            predicted_noise = model(x_t, t)

            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            wandb.log(
                {"batch_loss": loss.item(), "epoch": epoch, "global_step": global_step}
            )

            pbar.set_postfix(MSE=f"{loss.item():.4f}")
            global_step += 1

        avg_epoch_loss = epoch_loss / len(dataloader)
        wandb.log(
            {
                "epoch_loss": avg_epoch_loss,
                "epoch": epoch,
            }
        )

        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}_{time.time()}.pt")

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                sampled_images = diffusion.sample(model, n=16, labels=None)
                torchvision.utils.save_image(
                    sampled_images, f"samples/epoch_{epoch}_{time.time()}.png"
                )

                images = wandb.Image(
                    sampled_images, caption=f"Generated samples at epoch {epoch}"
                )
                wandb.log({"generated_samples": images})

                # Also log a grid of noised images at different timesteps
                noise_grid = create_noise_progression(
                    diffusion, images[0].unsqueeze(0), model
                )
                wandb.log(
                    {
                        "noise_progression": wandb.Image(
                            noise_grid, caption=f"Noise progression at epoch {epoch}"
                        )
                    }
                )

    wandb.finish()


def create_noise_progression(diffusion, image, model, num_steps=8):
    """Creates a grid showing the denoising process."""
    model.eval()
    with torch.no_grad():
        t = torch.linspace(0, diffusion.noise_steps - 1, num_steps).long().to(DEVICE)

        noised_images = []
        for dt in t:
            noised, _ = diffusion.noise_images(image, torch.tensor([dt]).to(DEVICE))
            noised_images.append(noised)

        grid = torchvision.utils.make_grid(
            torch.cat(noised_images, dim=0),
            nrow=num_steps,
            normalize=True,
            value_range=(-1, 1),
        )

        return grid


if __name__ == "__main__":
    train(num_epochs=10, batch_size=32)
