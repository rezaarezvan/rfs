import torch
import torch.optim as optim
import torch.nn.functional as F

from rfs import DEVICE
from rfs.models.vae import VAE
from rfs.data.dataloaders import get_mnist_loader

model = VAE(input_channels=1, latent_dim=4).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x, mu, log_var):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss


train_loader = get_mnist_loader(32, train=True)

for epoch in range(10):
    model.train()
    train_loss = 0
    for data, _ in train_loader:  # Assuming labels are not needed
        data = data.to(DEVICE)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch {epoch}: Average loss {
          train_loss / len(train_loader.dataset)}')

