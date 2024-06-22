# training.py

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss

# Import our model and parameters
from model import ViT
from parameters import *
from dataset import train_loader

# Define the device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model and move it to the device
model = ViT().to(device)

# Define the loss function and the optimizer
criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop

def train():
    model.train()  # Ensure the model is in training mode
    total_loss = 0
    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # print statistics
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] average loss: %.3f' % (epoch + 1, i + 1, total_loss / 2000))
                total_loss = 0

    print('Finished Training')
    # save the model after training
    torch.save(model.state_dict(), './model/model.pth')