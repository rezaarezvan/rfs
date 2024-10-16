import time
import torch
import torch.optim as optim
import torch.nn.functional as F

from rfs import DEVICE
from rfs.models.vae import VAE
from rfs.utils.vae_utils import visualize_reconstructions, visualize_samples, visualize_latent_space
from rfs.data.dataloaders import get_mnist_loader

model = VAE(input_channels=1, latent_dim=16).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


def loss_function(recon_x, x, mu, log_var, kl_weight=1e-3):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (recon_loss + kl_loss*kl_weight) / x.size(0)


train_loader = get_mnist_loader(32, train=True)
conditional = False

current_loss = float('inf')

for epoch in range(10):
    model.train()
    train_loss = 0
    for image, labels in train_loader:
        image, labels = image.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(
            image, labels) if conditional else model(image)
        loss = loss_function(recon_batch, image, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        if current_loss > loss.item():
            current_loss = loss.item()
            torch.save(model.state_dict(),
                       f'result/vae_{epoch}_{int(time.time())}.pth')
        optimizer.step()
    print(f'Epoch {epoch}: Average loss {
          train_loss / len(train_loader.dataset)}')


'''
For visualization:

model.load_state_dict(torch.load('result/vae_5_1729094804.pth'))
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
'''
