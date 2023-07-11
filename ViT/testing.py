# testing.py

import torch

# Import our model and parameters
from model import ViT
from dataset import test_loader
from parameters import *

# Define the device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model and move it to the device
model = ViT().to(device)

# Load the trained model
model.load_state_dict(torch.load('./model/model.pth'))
model.eval()

def test():
# Initialize counters
    correct = 0
    total = 0

# No need to track gradients for testing, so wrap in
# no_grad to save memory
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on test images: %d %%' %
          (100 * correct / total))
