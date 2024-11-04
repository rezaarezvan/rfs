import os
import time
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
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
            transforms.Lambda(lambda t: t.float()),
        ]
    )
    dataset = MNIST(root="./data", train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_vae():
    """Load pretrained VAE"""
    vae = VAE(input_channels=1, latent_dim=32).to(DEVICE)
    vae.load_state_dict(torch.load("result/vae.pth"))
    vae.eval()
    return vae


def sample_and_save(ema_model, vae, args, train_steps, in_channels):
    """Generate and save samples"""
    ema_model.eval()
    with torch.no_grad():
        if vae is None:
            samples = torch.randn(16, in_channels, 28, 28).to(DEVICE)
            samples = p_sample_loop(
                ema_model, samples.shape, noise=samples, clip_denoised=True
            )
        else:
            latent_samples = torch.randn(16, in_channels, 6, 6).to(DEVICE)
            latent_samples = p_sample_loop(
                ema_model,
                latent_samples.shape,
                noise=latent_samples,
                clip_denoised=True,
            )
            latent_flat = latent_samples.view(latent_samples.size(0), -1)[:, :32]
            samples = vae.decode(latent_flat)

        vutils.save_image(
            samples,
            f"{args.results_dir}/samples/sample_{train_steps:07d}.png",
            normalize=True,
            nrow=4,
        )


def train(args):
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(f"{args.results_dir}/samples", exist_ok=True)
    os.makedirs(f"{args.results_dir}/checkpoints", exist_ok=True)

    if args.vae:
        vae = load_vae()
        in_channels = 1
        spatial_size = 6
    else:
        in_channels = 1
        spatial_size = 28

    model = DiT(
        input_size=spatial_size,
        patch_size=2,
        in_channels=in_channels,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
    ).to(DEVICE)

    diffusion = create_diffusion()

    ema = deepcopy(model).to(DEVICE)
    for param in ema.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.03, betas=(0.9, 0.999)
    )

    train_loader = get_mnist_loader(args.batch_size, train=True)

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for x, _ in pbar:
            x = x.to(DEVICE)

            if vae is not None:
                with torch.no_grad():
                    mu, log_var = vae.encode(x)
                    z = vae.reparameterize(mu, log_var)
                    z_padded = F.pad(z, (0, 4), mode="constant", value=0)
                    x = z_padded.view(
                        z.size(0), in_channels, spatial_size, spatial_size
                    )

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=DEVICE)

            loss_dict = training_losses(model, x, t)
            loss = loss_dict["loss"]

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

                sample_and_save(ema, vae, args, train_steps, in_channels)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--results-dir", type=str, default="results/dit_mnist")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--vae", action="store_true", help="Train in VAE latent space")
    args = parser.parse_args()

    train(args)
