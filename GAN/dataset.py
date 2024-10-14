from parameters import Params
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=Params.batch_size,
        shuffle=True
    )

    return train_loader
