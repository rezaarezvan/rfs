import torchvision.transforms as transforms

from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader


def get_mnist_loader(batch_size, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = MNIST(root='./data', train=train,
                    download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def get_cifar10_loader(batch_size, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = CIFAR10(root='./data', train=train,
                      download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
