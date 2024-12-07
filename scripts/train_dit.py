import os
import time
import torch
import wandb

import torchvision.transforms as transforms

from tqdm import tqdm
from copy import deepcopy
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from rfs import DEVICE
from rfs.models.dit import DiT
from rfs.models.vae import VAE
from rfs.utils.dit_utils import create_diffusion, training_losses, p_sample_loop


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Update EMA parameters"""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(param.data, alpha=1 - decay)


def get_mnist_loader(batch_size, train=True):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )
    dataset = MNIST(root="./data", train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_vae():
    """Load pretrained VAE"""
    vae = VAE(input_channels=1, latent_dim=32).to(DEVICE)
    vae.load_state_dict(torch.load("result/vae.pth", map_location=DEVICE))
    vae.eval()
    return vae


def train(args):
    wandb.init(
        project="diffusion_experiment", name="dit_mnist_run1", config=args.__dict__
    )
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(f"{args.results_dir}/checkpoints", exist_ok=True)
    train_loader = get_mnist_loader(args.batch_size, train=True)

    if args.vae:
        vae = load_vae()
        input_size = 4
        in_channels = 2
    else:
        vae = None
        input_size = 28
        in_channels = 1

    model = DiT(
        input_size=input_size,
        in_channels=in_channels,
        patch_size=2,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        learn_sigma=False,
    ).to(DEVICE)

    diffusion = create_diffusion()
    ema = deepcopy(model).to(DEVICE)
    for param in ema.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.03, betas=(0.9, 0.999)
    )

    train_steps = 0
    start_epoch = 0

    if args.load:
        print(f"Loading checkpoint from {args.load}")
        checkpoint = torch.load(args.load, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_steps = checkpoint["train_steps"]
        steps_per_epoch = len(train_loader)
        start_epoch = train_steps // steps_per_epoch
        print(f"Resuming from step {train_steps} (epoch {start_epoch})")

    log_steps = 0
    running_loss = 0
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for x, _ in pbar:
            x = x.to(DEVICE)
            batch_size = x.size(0)

            if vae is not None:
                with torch.no_grad():
                    mu, log_var = vae.encode(x)
                    latents = vae.reparameterize(mu, log_var)
                    x = latents.view(batch_size, in_channels, input_size, input_size)

            t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=DEVICE)

            loss = training_losses(model, x, t, diffusion)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            update_ema(ema, model)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                steps_per_sec = log_steps / (time.time() - start_time)
                avg_loss = running_loss / log_steps
                wandb.log({"train_loss": avg_loss}, step=train_steps)
                pbar.set_postfix(
                    {"loss": f"{avg_loss:.4f}", "steps/sec": f"{steps_per_sec:.2f}"}
                )

                running_loss = 0
                log_steps = 0
                start_time = time.time()

            if train_steps % args.sample_every == 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_steps": train_steps,
                }
                torch.save(
                    checkpoint,
                    f"{args.results_dir}/checkpoints/checkpoint_{train_steps:07d}.pt",
                )

                samples = p_sample_loop(
                    model, (16, in_channels, input_size, input_size)
                )
                samples = samples.cpu().numpy()
                for sample in samples:
                    ch1 = ((sample[..., 0] + 1) / 2.0).clip(0, 1)
                    ch2 = ((sample[..., 1] + 1) / 2.0).clip(0, 1)
                    wandb.log(
                        {"channel_1": wandb.Image(ch1), "channel_2": wandb.Image(ch2)}
                    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--results-dir", type=str, default="result/dit_mnist")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--vae", action="store_true", help="Train in VAE latent space")
    parser.add_argument(
        "--load", type=str, help="Path to checkpoint to resume training from"
    )
    args = parser.parse_args()

    train(args)
