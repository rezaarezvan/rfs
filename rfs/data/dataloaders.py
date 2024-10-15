import os
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image


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


def get_data_from_path(args):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(
            args.image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(
        args.dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def get_oxford_pet_dataloaders(batch_size):
    class OxfordPetDataset(Dataset):
        def __init__(self, root, transforms_img=None, transforms_mask=None):
            self.root = root
            self.transforms_img = transforms_img
            self.transforms_mask = transforms_mask
            self.images = []
            self.masks = []

            images_dir = os.path.join(root, 'images')
            masks_dir = os.path.join(root, 'annotations/trimaps')
            for file in os.listdir(images_dir):
                if file.endswith('.jpg'):
                    self.images.append(os.path.join(images_dir, file))
                    mask_file = file.replace('.jpg', '.png')
                    self.masks.append(os.path.join(masks_dir, mask_file))

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = self.images[idx]
            mask_path = self.masks[idx]

            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path)

            mask = np.array(mask)
            mask = np.where(mask == 2, 1, 0)
            mask = Image.fromarray(mask.astype('uint8'))

            if self.transforms_img:
                image = self.transforms_img(image)
            if self.transforms_mask:
                mask = self.transforms_mask(mask)
                mask = mask.squeeze(0)

            return image, mask

    dataset = OxfordPetDataset(
        root='data/oxford-iiit-pet',
        transforms_img=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]),
        transforms_mask=transforms.Compose([
            transforms.Resize((128, 128), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader
