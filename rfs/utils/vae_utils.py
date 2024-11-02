import os
import time
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from rfs import DEVICE
from sklearn.manifold import TSNE


def visualize_reconstructions(model, data_loader, num_images=8):
    model.eval()
    with torch.no_grad():
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if model.conditional:
            recon_images, _, _ = model(images, labels)
        else:
            recon_images, _, _ = model(images)

        images = images.cpu().numpy()
        recon_images = recon_images.cpu().numpy()

        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
        for i in range(num_images):
            axes[0, i].imshow(images[i].squeeze(), cmap="gray")
            axes[0, i].axis("off")

            axes[1, i].imshow(recon_images[i].squeeze(), cmap="gray")
            axes[1, i].axis("off")


def visualize_samples(model, num_samples=8, labels=None):
    model.eval()
    with torch.no_grad():
        if labels is not None:
            labels = torch.tensor(labels).to(DEVICE)
        else:
            labels = torch.randint(0, model.num_classes, (num_samples,)).to(DEVICE)
        samples = model.sample(num_samples, labels)
        samples = samples.cpu().numpy()

        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
        for i in range(num_samples):
            axes[i].imshow(samples[i].squeeze(), cmap="gray")
            axes[i].axis("off")
            if labels is not None:
                axes[i].set_title(f"Label: {labels[i].item()}")


def visualize_latent_space(model, data_loader):
    model.eval()
    all_mu = []
    all_labels = []
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)
            mu, _ = (
                model.encode(data, labels) if model.conditional else model.encode(data)
            )
            all_mu.append(mu.cpu())
            all_labels.append(labels.cpu())
    all_mu = torch.cat(all_mu).numpy()
    all_labels = torch.cat(all_labels).numpy()

    tsne = TSNE(n_components=2)
    mu_2d = tsne.fit_transform(all_mu)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=mu_2d[:, 0], y=mu_2d[:, 1], hue=all_labels, palette="tab10", legend="full"
    )
    plt.title("Latent Space Visualization")


@torch.no_grad()
def save_latents(model, data_loader, path):
    """Save VAE latents for all images in the dataset"""
    model.eval()
    latents = []
    for data, labels in data_loader:
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        mu, log_var = (
            model.encode(data, labels) if model.conditional else model.encode(data)
        )
        latent = model.reparameterize(mu, log_var)
        latents.append(latent.cpu())

    latents = torch.cat(latents, dim=0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(latents, f"{path}_{time.time()}.pt")
