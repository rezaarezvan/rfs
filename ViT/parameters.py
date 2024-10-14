from dataclasses import dataclass


@dataclass
class DatasetParams:
    img_size: int = 32  # CIFAR-10 dataset image size
    patch_size: int = 4  # Patch size to break images into
    num_classes: int = 10  # Number of CIFAR-10 dataset classes


@dataclass
class ModelParams:
    embed_dim: int = 64  # Patch embedding dimension
    num_heads: int = 4  # Number of attention heads
    num_layers: int = 4  # Number of transformer layers


@dataclass
class TrainingParams:
    batch_size: int = 64  # Batch size for training
    learning_rate: float = 1e-3  # Optimizer learning rate
    num_epochs: int = 10  # Number of epochs for training
