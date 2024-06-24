import torch
from .config import *

SEED = 42069
device = 'cuda' if torch.cuda.is_available() and get_config()['device'] == 'cuda' else 'cpu'
DEVICE = torch.device(device)
