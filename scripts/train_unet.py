from torch import optim, nn
from rfs import DEVICE
from rfs.models.unet import UNet
from rfs.data.dataloaders import get_oxford_pet_dataloaders


def main():
    device = DEVICE
    model = UNet(n_channels=3, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_loader, val_loader = get_oxford_pet_dataloaders(batch_size=8)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss / len(train_loader):.4f}"
        )


if __name__ == "__main__":
    main()
