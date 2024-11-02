import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = ["DEVICE"]
