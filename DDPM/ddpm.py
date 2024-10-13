import torch
import logging
import torch.nn as nn

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from modules import UNet
from utils import get_data, setup_logging, save_images

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.INFO, datefmt='%I:%M:%S')


class Diffusion:
    def __init__(self, num_noise_steps=2, beta_start=1e-4, beta_end=2*1e-2, img_size=64):
        self.num_noise_steps = num_noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.num_noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(
            1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)

        return sqrt_alpha_hat*x + sqrt_one_minus_alpha_hat*noise, noise

    def sample_timestemps(self, n):
        return torch.randint(low=1, high=self.num_noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images...")
        model.eval()

        with torch.no_grad():
            x = torch.randn(
                (n, 3, self.img_size, self.img_size)).to(self.device)

            for i in tqdm(reversed(range(1, self.num_noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.rand_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat)))
                                             * predicted_noise) + torch.sqrt(beta) * noise

                model.train()
                x = (x.clamp(-1, 1) + 1) / 2
                x = (x * 255).type(torch.uint8)

                return x


def train(args):
    setup_logging(args.run_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size)
    logger = SummaryWriter(f'runs/{args.run_name}')
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch + 1}/{args.epochs}...")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timestemps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch*l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, f"results/{args.run_name}/{epoch}.png")
        torch.save(model.state_dict(), f"models/{args.run_name}/{epoch}.pt")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Unconditional"
    args.epochs = 1
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = "datasets/unconditional"
    args.lr = 3e-4
    train(args)


if __name__ == "__main__":
    main()
