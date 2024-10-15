import torch
import argparse
import logging
import torch.nn as nn

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from rfs import DEVICE
from rfs.models.diffusion import Diffusion, UNet, UNetConditional, EMA
from rfs.utils.logging import setup_logging
from rfs.utils.image_utils import save_images
from rfs.data.dataloaders import get_data_from_path


def train(args):
    setup_logging(args.run_name)
    device = DEVICE
    dataloader = get_data_from_path(args)
    model = UNet().to(device)
    # model = UNetConditional(num_classes=args.nums_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size)
    logger = SummaryWriter(f'runs/{args.run_name}')
    l = len(dataloader)
    # ema = EMA(beta=0.995)
    # ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch + 1}/{args.epochs}...")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            t = diffusion.sample_timestemps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            # if np.random.random() < 0.1:
            #     labels = None
            predicted_noise = model(x_t, t)
            # predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch*l + i)

        sampled_images = diffusion.sample(
            model, n=images.shape[0], labels=None, cfg_scales=0)
        # ema_sampled_images = diffusion.sample(ema_model, n=images.shape[0])
        save_images(sampled_images, f"results/{args.run_name}/{epoch}.png")
        # save_images(ema_sampled_images, f"results/{args.run_name}/ema_{epoch}.png
        torch.save(model.state_dict(), f"models/{args.run_name}/{epoch}.pt")
        # torch.save(ema_model.state_dict(), f"models/{args.run_name}/ema_{epoch}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='DDPM_Unconditional')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--dataset_path', type=str,
                        default='datasets/unconditional')
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
