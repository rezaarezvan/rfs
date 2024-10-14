import torch
import torch.optim as optim

from model import ViT
from dataset import train_loader
from torch.nn import CrossEntropyLoss
from parameters import TrainingParams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViT().to(device)

criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=TrainingParams.learning_rate)


def train():
    model.train()
    total_loss = 0
    for epoch in range(TrainingParams.num_epochs):
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % 2000 == 1999:
                print('[%d, %5d] average loss: %.3f' %
                      (epoch + 1, i + 1, total_loss / 2000))
                total_loss = 0

    print('Finished Training')
    torch.save(model.state_dict(), './model/model.pth')
