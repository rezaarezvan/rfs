import os
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


def train(args):
    wandb.init(
        project="diffusion_experiment", name="dppm_mnist_run1", config=args.__dict__
    )

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(f"{args.results_dir}/checkpoints", exist_ok=True)
    train_loader = get_mnist_loader(args.batch_size, train=True)

    model = UNet(num_in_channels=1, num_out_channels=1).to(DEVICE)
    diffusion = DDPM(img_size=28)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.03, betas=(0.9, 0.999)
    )

    mse = nn.MSELoss()
    wandb.watch(model, log_freq=100)

    train_steps = 0
    start_epoch = 0

    if args.load:
        print(f"Loading model from {args.load}")
        checkpoint = torch.load(args.load, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_steps = checkpoint["train_steps"]
        steps_per_epoch = len(train_loader)
        start_epoch = train_steps // steps_per_epoch
        print(f"Resuming training from epoch {start_epoch}")

    log_steps = 0
    running_loss = 0
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for images, _ in pbar:
            images = images.to(DEVICE)

            t = diffusion.sample_timesteps(images.shape[0]).to(DEVICE)

            x_t, noise = diffusion.noise_images(images, t)

            predicted_noise = model(x_t, t)

            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                steps_per_sec = log_steps / (time.time() - start_time)
                avg_loss = running_loss / log_steps
                wandb.log({"train_loss": avg_loss}, step=train_step)
                pbar.set_postfix(
                    {"loss": f"{avg_loss:.4f}", "steps/sec": f"{steps_per_sec:.2f}"}
                )

                running_loss = 0
                log_steps = 0
                start_time = time.time()

        if train_steps % args.sample_every == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_steps": train_steps,
            }

            torch.save(
                checkpoint,
                f"{args.results_dir}/checkpoints/checkpoint_{train_steps:07d}.pt",
            )

            model.eval()
            with torch.no_grad():
                sampled_images = diffusion.sample(model, n=16, labels=None)
                torchvision.utils.save_image(
                    sampled_images, f"samples/epoch_{epoch}_{time.time()}.png"
                )

                images = wandb.Image(
                    sampled_images,
                    caption=f"Generated samples at epoch {epoch}",
                )
                wandb.log({"generated_samples": images})

                # Also log a grid of noised images at different timesteps
                noise_grid = create_noise_progression(
                    diffusion, images[0].unsqueeze(0), model
                )
                wandb.log(
                    {
                        "noise_progression": wandb.Image(
                            noise_grid,
                            caption=f"Noise progression at epoch {epoch}",
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--results-dir", type=str, default="result/ddpm_mnist")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument(
        "--load", type=str, help="Path to checkpoint to resume training from"
    )

    args = parser.parse_args()

    train(args)
