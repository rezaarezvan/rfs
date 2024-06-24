import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.model.input_dim, cfg.model.hidden_dim)
        self.fc2 = nn.Linear(cfg.model.hidden_dim, cfg.model.output_dim)
        self.dropout = nn.Dropout(cfg.model.dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_model(cfg):
    return SimpleNN(cfg)
