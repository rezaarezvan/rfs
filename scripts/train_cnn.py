import torch

from rfs import DEVICE
from rfs.models.cnn import SimpleCNN
from rfs.trainers.trainer import Trainer
from rfs.data.dataloaders import get_mnist_loader


def main():
    device = DEVICE
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, device)

    train_loader = get_mnist_loader(batch_size=64, train=True)
    val_loader = get_mnist_loader(batch_size=64, train=False)

    for epoch in range(10):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)
        print(
            f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={
                val_loss:.4f
            }, Val Acc={val_acc:.2f}%"
        )


if __name__ == "__main__":
    main()
