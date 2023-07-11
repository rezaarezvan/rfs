# dataset.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import the parameters we defined
from parameters import IMG_SIZE, BATCH_SIZE

# Normalizing the images to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and load the training data
train_set = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

# Download and load the test data
test_set = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
