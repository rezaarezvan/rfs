import torch
import torch.nn as nn
import torchvision.transforms as transforms


from tqdm import tqdm
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from rfs import DEVICE
from rfs.models.diffusion import DiT
from rfs.utils.dit_utils import cosine_alphas_bar


def get_mnist_loader(batch_size, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
        transforms.Lambda(lambda t: t.float()),
    ])

    dataset = MNIST(root='./data', train=train,
                    download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(num_epochs=10, batch_size=32):
    dataloader = get_mnist_loader(batch_size=batch_size)
    time_steps = 1000
    patch_size = 2

    model = DiT(image_size=28, channels_in=1, patch_size=patch_size, hidden_size=128,
                num_features=128, num_layers=3, num_heads=4).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-2)

    scaler = torch.cuda.amp.GradScaler()

    alphas = torch.flip(cosine_alphas_bar(time_steps), (0,)).to(DEVICE)

    loss_log = []
    mean_loss = 0

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(dataloader)
        pbar.set_description(f"Epoch {epoch}")
        mean_loss = 0

        for batch in pbar:
            if isinstance(batch, (tuple, list)):
                latents, _ = batch
            else:
                latents = batch

            latents = latents.float().to(DEVICE)

            bs = latents.shape[0]
            rand_index = torch.randint(time_steps, (bs,), device=DEVICE)
            random_sample = torch.randn_like(latents)
            alpha_batch = alphas[rand_index].reshape(bs, 1, 1, 1)

            noise_input = alpha_batch.sqrt() * latents +\
                (1 - alpha_batch).sqrt() * random_sample

            with torch.cuda.amp.autocast():
                latent_pred = model(noise_input, rand_index)
                loss = nn.functional.mse_loss(latent_pred, latents)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_log.append(loss.item())
            mean_loss += loss.item()

        torch.save({'epoch': epoch + 1,
                    'train_data_logger': loss_log,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, "latent_dit.pt")


if __name__ == "__main__":
    train(num_epochs=10, batch_size=32)
