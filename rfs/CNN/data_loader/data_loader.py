from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.data.mean, cfg.data.std)
    ])

    train_dataset = datasets.MNIST(cfg.paths.data_dir, train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(cfg.paths.data_dir, train=False, transform=transform)
    test_dataset = datasets.MNIST(cfg.paths.data_dir, train=False, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
