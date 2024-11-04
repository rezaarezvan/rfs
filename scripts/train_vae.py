import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from rfs import DEVICE
from rfs.models.vae import VAE
from rfs.utils.vae_utils import (
    visualize_reconstructions,
    visualize_samples,
    visualize_latent_space,
    save_latents,
)
from rfs.data.dataloaders import get_mnist_loader

model = VAE(input_channels=1, latent_dim=32).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", factor=0.5, patience=5, verbose=True
)


def loss_function(recon_x, x, mu, log_var, kl_weight=0.001):
    recon_loss = (
        F.binary_cross_entropy(recon_x, x, reduction="none").sum(dim=(1, 2, 3)).mean()
    )

    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


train_loader = get_mnist_loader(128, train=True)
val_loader = get_mnist_loader(128, train=False)

conditional = False
num_epochs = 100
best_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0

    kl_weight = min((epoch + 1) / 50, 1.0) * 0.001

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

    for batch_idx, (image, labels) in enumerate(pbar):
        image, labels = image.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        recon_batch, mu, log_var = model(image, labels) if conditional else model(image)
        loss, recon_loss, kl_loss = loss_function(
            recon_batch, image, mu, log_var, kl_weight
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()

        pbar.set_postfix({"loss": loss.item()})

    avg_train_loss = train_loss / len(train_loader.dataset)
    scheduler.step(avg_train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for image, labels in val_loader:
            image, labels = image.to(DEVICE), labels.to(DEVICE)
            recon_batch, mu, log_var = (
                model(image, labels) if conditional else model(image)
            )
            val_loss += loss_function(recon_batch, image, mu, log_var).item()

    avg_val_loss = val_loss / len(val_loader.dataset)

    print(f"Epoch {epoch}: Train Loss = {
          avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), f"result/vae_best_{int(time.time())}.pth")

    if epoch % 10 == 0:
        visualize_reconstructions(model, val_loader)
        plt.savefig(f"result/reconstructions_{epoch}_{int(time.time())}.png")
        plt.close()


print("Generating latents for all images in the dataset")
save_latents(model, train_loader, "result/latents.pth")

"""
For visualization:

model.load_state_dict(torch.load('result/vae.pth'))
conditional = False

test_loader = get_mnist_loader(32, train=False)
for epoch in range(2):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for image, labels in test_loader:
            image, labels = image.to(DEVICE), labels.to(DEVICE)
            recon_batch, mu, log_var = model(
                image, labels) if conditional else model(image)
            test_loss += loss_function(recon_batch, image, mu, log_var).item()
        visualize_reconstructions(model, test_loader)
    print(f'Epoch {epoch}: Average test loss {
          test_loss / len(test_loader.dataset)}')

visualize_samples(model, num_samples=8, labels=[0, 1, 2, 3, 4, 5, 6, 7])
latent_loader = get_mnist_loader(256, train=False)
visualize_latent_space(model, latent_loader)
"""
