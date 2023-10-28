import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from parameters import DatasetParams, TrainingParams

class DatasetLoader:
    def __init__(self, root='./data', download=True, batch_size=64):
        self.root = root
        self.download = download
        self.batch_size = batch_size
        self.transform = self.get_transform()
        self.train_set = self.get_dataset(train=True)
        self.test_set = self.get_dataset(train=False)
        self.train_loader = self.get_dataloader(self.train_set, shuffle=True)
        self.test_loader = self.get_dataloader(self.test_set, shuffle=False)

    @staticmethod
    def get_transform():
        # Normalizing the images to [-1, 1]
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_dataset(self, train):
        # Using CIFAR10 dataset for training and testing
        return datasets.CIFAR10(
            root=self.root, train=train, download=self.download, transform=self.transform)

    def get_dataloader(self, dataset, shuffle):
        # DataLoader for batching, shuffling and loading data in parallel
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

# Parameters for dataset and training
dataset_params = DatasetParams()
training_params = TrainingParams()

# Initialize DatasetLoader
data_loader = DatasetLoader(dataset_params.root, dataset_params.download, training_params.batch_size)
train_loader = data_loader.train_loader
test_loader = data_loader.test_loader