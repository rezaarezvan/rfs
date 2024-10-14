import torch

from model import ViT
from dataset import test_loader
from parameters import ModelParams, DatasetParams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_params = ModelParams()
model = ViT(d_model=model_params.embed_dim, nhead=model_params.num_heads,
            num_layers=model_params.num_layers, num_classes=DatasetParams().num_classes).to(device)

try:
    checkpoint = torch.load('./model/model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
except FileNotFoundError:
    print("Trained model not found. Please train the model first.")
    exit()


def test():
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on test images: {accuracy:.2f} %')
