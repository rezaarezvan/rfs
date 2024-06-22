import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 512)
        self.fc2 = nn.Linear(512, num_outputs)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
