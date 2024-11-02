import copy
import torch

from rfs import DEVICE
from tqdm import trange


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
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_bar = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    return alphas_bar[:timesteps]


def noise_from_x0(curr_img, img_pred, alpha):
    return (curr_img - alpha.sqrt() * img_pred) / ((1 - alpha).sqrt() + 1e-4)


def cold_diffuse(diffusion_model, sample_in, total_steps, start_step=0):
    diffusion_model.eval()
    bs = sample_in.shape[0]
    alphas = torch.flip(cosine_alphas_bar(total_steps), (0,)).to(DEVICE)
    random_sample = copy.deepcopy(sample_in)
    with torch.no_grad():
        for i in trange(start_step, total_steps - 1):
            index = (i * torch.ones(bs, device=sample_in.device)).long()

            img_output = diffusion_model(random_sample, index)

            noise = noise_from_x0(random_sample, img_output, alphas[i])
            x0 = img_output

            rep1 = alphas[i].sqrt() * x0 + (1 - alphas[i]).sqrt() * noise
            rep2 = alphas[i + 1].sqrt() * x0 + (1 - alphas[i + 1]).sqrt() * noise

            random_sample += rep2 - rep1

        index = ((total_steps - 1) * torch.ones(bs, device=sample_in.device)).long()
        img_output = diffusion_model(random_sample, index)

    return img_output
