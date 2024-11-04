import os
import time
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms

from tqdm import tqdm
from rfs import DEVICE
from rfs.models.dit import DiT
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from rfs.utils.dit_utils import cosine_alphas_bar, cold_diffuse


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


def train(num_epochs=100, batch_size=128):
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    train_loader = get_mnist_loader(batch_size=batch_size)

    model = DiTLatent(
        image_size=28,
        channels_in=1,
        patch_size=2,
        hidden_size=768,
        num_heads=12,
        num_layers=12,
        num_classes=None,
    ).to(DEVICE)

    GLOBAL_BATCH_SIZE = 256
    ACCUMULATION_STEPS = max(GLOBAL_BATCH_SIZE // batch_size, 1)
    GRADIENT_CLIP_NORM = 1.0
    time_steps = 1000

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.03, betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5
    )

    scaler = torch.cuda.amp.GradScaler()
    alphas = torch.flip(cosine_alphas_bar(time_steps), (0,)).to(DEVICE)

    loss_log = []
    best_loss = float("inf")
    start_epoch = 0

    checkpoint_path = f"checkpoints/latest_dit_{int(time.time())}.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        loss_log = checkpoint["loss_log"]
        best_loss = checkpoint["best_loss"]
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, (tuple, list)):
                images, _ = batch
            else:
                images = batch

            images = images.float().to(DEVICE)
            bs = images.shape[0]

            t = torch.randint(time_steps, (bs,), device=DEVICE)
            noise = torch.randn_like(images)
            alpha_t = alphas[t].reshape(bs, 1, 1, 1)
            noisy_images = alpha_t.sqrt() * images + (1 - alpha_t).sqrt() * noise

            with torch.cuda.amp.autocast():
                pred = model(noisy_images, t)
                loss = nn.functional.mse_loss(pred, images)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * ACCUMULATION_STEPS
            pbar.set_postfix({"loss": loss.item() * ACCUMULATION_STEPS})
            loss_log.append(loss.item() * ACCUMULATION_STEPS)

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} average loss: {avg_loss:.6f}")

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss_log": loss_log,
                "best_loss": best_loss,
            },
            checkpoint_path,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "loss": avg_loss,
                },
                f"checkpoints/best_dit_{int(time.time())}.pt",
            )

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                samples = torch.randn(16, 1, 28, 28).to(DEVICE)
                samples = cold_diffuse(model, samples, total_steps=100)
                vutils.save_image(
                    samples,
                    f"results/dit_samples_epoch_{epoch +
                                                 1}_{int(time.time())}.png",
                    normalize=True,
                    nrow=4,
                )


if __name__ == "__main__":
    train(num_epochs=10, batch_size=128)
