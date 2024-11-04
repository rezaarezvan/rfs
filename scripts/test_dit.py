import torch
import torch.nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from rfs import DEVICE
from rfs.models.vae import VAE
from rfs.models.dit import DiT
from rfs.utils.dit_utils import cold_diffuse
from rfs.utils.vae_utils import (
    visualize_reconstructions,
    visualize_samples,
    visualize_latent_space,
)


def load_models(vae_path="result/vae.pth", dit_path="result/latest_dit_1730564606.pt"):
    vae = VAE(input_channels=1, latent_dim=32).to(DEVICE)
    vae.load_state_dict(torch.load(vae_path))
    vae.eval()

    dit = DiTLatent(
        image_size=28,
        channels_in=1,
        patch_size=2,
        hidden_size=768,
        num_heads=12,
        num_layers=12,
        num_classes=None,
    ).to(DEVICE)

    dit.load_state_dict(torch.load(dit_path)["model_state_dict"])
    dit.eval()

    return vae, dit


def get_test_loader(batch_size=128):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)]
    )
    dataset = MNIST(root="./data", train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def visualize_vae_results(vae, test_loader):
    print("Visualizing VAE Results...")

    visualize_reconstructions(vae, test_loader)
    plt.savefig("vae_reconstructions.png")
    plt.close()

    visualize_samples(vae, num_samples=8)
    plt.savefig("vae_samples.png")
    plt.close()

    visualize_latent_space(vae, test_loader)
    plt.savefig("vae_latent_space.png")
    plt.close()


def visualize_dit_results(dit, num_samples=8):
    print("Visualizing DiT Results...")

    with torch.no_grad():
        noise = torch.randn(num_samples, 1, 28, 28).to(DEVICE)

        total_steps = 100
        denoised = cold_diffuse(dit, noise, total_steps)

        denoised = (denoised + 1) / 2
        vutils.save_image(
            denoised,
            "dit_samples.png",
            nrow=4,
            normalize=False,
        )


def test_vae_dit_pipeline(vae, dit, test_loader, num_samples=8):
    print("Testing VAE-DiT Pipeline...")
    with torch.no_grad():
        test_images, _ = next(iter(test_loader))
        test_images = test_images[:num_samples].to(DEVICE)

        direct_recon, _, _ = vae(test_images)

        z_mu, z_log_var = vae.encode(test_images)
        z = vae.reparameterize(z_mu, z_log_var)

        z_img = z.view(z.size(0), 1, 8, 4)
        z_img = torch.nn.functional.interpolate(
            z_img, size=(28, 28), mode="bicubic", align_corners=False
        )

        z_img = torch.tanh(z_img)

        denoised_z = cold_diffuse(dit, z_img, total_steps=100)

        denoised_z = torch.nn.functional.interpolate(
            denoised_z, size=(8, 4), mode="bicubic", align_corners=False
        )

        denoised_z = denoised_z.view(denoised_z.size(0), -1)

        final_output = vae.decode(denoised_z)

        vutils.save_image(z_img, "debug_z_img.png", normalize=True)
        vutils.save_image(
            denoised_z.view(denoised_z.size(0), 1, 8, 4),
            "debug_denoised_z.png",
            normalize=True,
        )

        test_images = (test_images + 1) / 2
        direct_recon = (direct_recon + 1) / 2
        final_output = (final_output + 1) / 2

        comparison = torch.cat([test_images, direct_recon, final_output])

        vutils.save_image(
            comparison,
            "combined_pipeline_test.png",
            nrow=num_samples,
            normalize=False,
        )


def main():
    print("Loading models...")
    vae, dit = load_models()
    test_loader = get_test_loader()

    visualize_vae_results(vae, test_loader)
    visualize_dit_results(dit)
    test_vae_dit_pipeline(vae, dit, test_loader)

    print("Done! Check the current directory for output images.")


if __name__ == "__main__":
    main()
