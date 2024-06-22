import os
import time
import torch
import numpy as np
import torch.nn as nn

from tqdm.auto import tqdm
from utils.config import get_config
from utils import DEVICE

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def train(model, train_loader, test_loader, optim, epochs=40, lossfn=nn.CrossEntropyLoss(), writer=None):
    config = get_config()
    model.to(DEVICE)
    global_step = 0
    best_accuracy = 0

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, leave=True)
        losses, accuracies = [], []
        for data, labels in loop:
            data, labels = data.to(DEVICE), labels.to(DEVICE)

            out = model(data)
            loss = lossfn(out, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            preds = out.argmax(dim=1)
            accuracy = (preds == labels).float().mean()

            losses.append(loss.item())
            accuracies.append(accuracy.item())

            if writer:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Accuracy/train', accuracy.item(), global_step)

            loop.set_description(f"Epoch {epoch+1}/{epochs}")
            loop.set_postfix(loss=loss.item(), accuracy=accuracy.item())

            global_step += 1

        avg_accuracy = evaluate(model, test_loader, writer, global_step)

        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            save_path = config['model_save_path']
            unique_path = os.path.join(save_path, f"{model.__class__.__name__}.{epoch}.{int(time.time())}.pth")

            save_model(model, unique_path)

    if writer:
        writer.close()


@torch.no_grad()
def evaluate(model, test_loader, writer=None, global_step=None):
    model.eval()
    accuracies = []

    for data, labels in tqdm(test_loader, leave=False):
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        outputs = model(data)
        preds = outputs.argmax(dim=1)
        accuracy = (preds == labels).float().mean()
        accuracies.append(accuracy.item())

    avg_accuracy = np.mean(accuracies)

    if writer and global_step is not None:
        writer.add_scalar('Accuracy/test', avg_accuracy, global_step)

    print(f"Validation Accuracy: {avg_accuracy:.4f}")
    return avg_accuracy
