import copy
import torch

from tqdm import tqdm
from rfs import DEVICE


def extract_patches(image_tensor, patch_size=8):
    bs, c, h, w = image_tensor.size()
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    unfolded = unfold(image_tensor)
    unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)
    return unfolded


def reconstruct_image(patch_sequence, image_shape, patch_size=8):
    """
    Reconstructs the original image tensor from a sequence of patches.
    Args:
        patch_sequence (torch.Tensor): Sequence of patches with shape
        BS x L x (C x patch_size x patch_size)
        image_shape (tuple): Shape of the original image tensor (bs, c, h, w).
        patch_size (int): Size of the patches used in extraction.
    Returns:
        torch.Tensor: Reconstructed image tensor.
    """
    bs, c, h, w = image_shape
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    unfolded_shape = (bs, num_patches_h, num_patches_w, patch_size, patch_size, c)
    patch_sequence = patch_sequence.view(*unfolded_shape)
    patch_sequence = patch_sequence.permute(0, 5, 1, 3, 2, 4).contiguous()
    reconstructed = patch_sequence.view(bs, c, h, w)
    return reconstructed


def cosine_alphas_bar(timesteps, s=0.008):
    """
    Compute the cumulative product of (1 - β_t) for t in [0, timesteps-1] using cosine scheduling.
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_bar = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    return alphas_bar[:timesteps]


def cold_diffuse(diffusion_model, sample_in, total_steps, start_step=0, cfg_scale=3.0):
    """
    DiT sampling process with classifier-free guidance support.
    Args:
        diffusion_model: The DiT model
        sample_in: Initial noise samples
        total_steps: Total number of denoising steps
        start_step: Starting timestep (for continuing partial sampling)
        cfg_scale: Classifier-free guidance scale (1.0 means no guidance)
    """
    diffusion_model.eval()
    bs = sample_in.shape[0]
    alphas = torch.flip(cosine_alphas_bar(total_steps), (0,)).to(DEVICE)
    random_sample = copy.deepcopy(sample_in)

    with torch.no_grad():
        for i in tqdm(range(start_step, total_steps - 1), desc="Sampling"):
            index = torch.full((bs,), i, device=DEVICE)

            predicted = diffusion_model(random_sample, index)

            alpha = alphas[i]
            alpha_next = (
                alphas[i + 1] if i < total_steps - 1 else torch.tensor(1.0).to(DEVICE)
            )

            sigma = torch.sqrt(
                (1 - alpha_next) / (1 - alpha) * (1 - alpha / alpha_next)
            )

            noise = (
                torch.randn_like(random_sample)
                if i < total_steps - 2
                else torch.zeros_like(random_sample)
            )

            x0_pred = (random_sample - torch.sqrt(1 - alpha) * predicted) / torch.sqrt(
                alpha
            )
            x0_pred = torch.clamp(x0_pred, -1, 1)

            random_sample = (
                torch.sqrt(alpha_next) * x0_pred
                + torch.sqrt(1 - alpha_next - sigma**2) * predicted
                + sigma * noise
            )

    return x0_pred
