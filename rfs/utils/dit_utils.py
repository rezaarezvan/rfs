import torch
import numpy as np

from tqdm import tqdm
from rfs import DEVICE
from types import SimpleNamespace


def get_beta_schedule_for_inference(betas, desired_count):
    """Adjust beta schedule length for inference."""
    orig_spacing = len(betas)
    subsample_steps = orig_spacing // desired_count

    new_betas = betas[::subsample_steps]

    if len(new_betas) > desired_count:
        new_betas = new_betas[:desired_count]
    elif len(new_betas) < desired_count:
        pad_size = desired_count - len(new_betas)
        new_betas = np.concatenate([new_betas, np.full(pad_size, new_betas[-1])])

    return new_betas


def create_diffusion(timestep_respacing=""):
    """
    Creates a noise schedule and diffusion process.
    """
    betas = get_named_beta_schedule("linear", 1000)
    if len(timestep_respacing) > 0:
        desired_count = int(timestep_respacing)
        new_betas = get_beta_schedule_for_inference(betas, desired_count)
        betas = new_betas

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

    return SimpleNamespace(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        sqrt_alphas_cumprod=np.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=np.sqrt(1.0 - alphas_cumprod),
        sqrt_recip_alphas_cumprod=np.sqrt(1.0 / alphas_cumprod),
        sqrt_recipm1_alphas_cumprod=np.sqrt(1.0 / alphas_cumprod - 1),
        num_timesteps=len(betas),
    )


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")


def extract(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    """
    res = torch.from_numpy(arr).to(DEVICE)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_model_output(model, x, t):
    """Get model output (predicted noise)"""
    return model(x, t)


def training_losses(model, x_start, t):
    """
    Compute training losses for a single timestep.
    """
    noise = torch.randn_like(x_start)
    x_t = (
        extract(diffusion.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        + extract(diffusion.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )
    model_output = get_model_output(model, x_t, t)

    loss = torch.nn.functional.mse_loss(model_output, noise, reduction="none")
    return {"loss": loss.mean()}


def p_sample(model, x, t, t_index):
    """
    Sample from the model at timestep t.
    """
    betas_t = extract(diffusion.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        diffusion.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(diffusion.sqrt_recip_alphas_cumprod, t, x.shape)

    model_output = get_model_output(model, x, t)
    pred_mean = sqrt_recip_alphas_t * (
        x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return pred_mean
    else:
        posterior_variance_t = extract(diffusion.betas, t, x.shape)
        noise = torch.randn_like(x)
        return pred_mean + torch.sqrt(posterior_variance_t) * noise


def p_sample_loop(model, shape, noise=None, clip_denoised=True, progress=True):
    """
    Generate samples from the model.
    """
    if noise is None:
        noise = torch.randn(*shape, device=DEVICE)
    x_start = noise

    indices = list(range(diffusion.num_timesteps))[::-1]
    if progress:
        indices = tqdm(indices, desc="Sampling")

    for i in indices:
        t = torch.tensor([i] * shape[0], device=DEVICE)
        with torch.no_grad():
            x_start = p_sample(model, x_start, t, i)
            if clip_denoised:
                x_start = torch.clamp(x_start, -1, 1)

    return x_start


diffusion = create_diffusion()
