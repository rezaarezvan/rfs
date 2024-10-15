import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


input_size = (128, 128)

transform_img = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
])

transform_mask = transforms.Compose([
    transforms.Resize(
        input_size, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.PILToTensor(),
])


class OxfordPetDataset(Dataset):
    def __init__(self, root, transforms_img=None, transforms_mask=None):
        self.root = root
        self.transforms_img = transforms_img
        self.transforms_mask = transforms_mask
        self.images = []
        self.masks = []

        images_dir = os.path.join(root, 'images')
        masks_dir = os.path.join(root, 'annotations/trimaps')
        for file in os.listdir(images_dir):
            if file.endswith('.jpg'):
                self.images.append(os.path.join(images_dir, file))
                mask_file = file.replace('.jpg', '.png')
                self.masks.append(os.path.join(masks_dir, mask_file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        mask = np.array(mask)
        mask = np.where(mask == 2, 1, 0)
        mask = Image.fromarray(mask.astype('uint8'))

        if self.transforms_img:
            image = self.transforms_img(image)
        if self.transforms_mask:
            mask = self.transforms_mask(mask)
            mask = mask.squeeze(0)

        return image, mask


dataset = OxfordPetDataset(
    root='data/oxford-iiit-pet',
    transforms_img=transform_img,
    transforms_mask=transform_mask
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 8

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else 'cpu')
criterion = nn.BCEWithLogitsLoss()

model = UNet(n_channels=3, n_classes=1)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 1

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

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device).float()
            outputs = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {epoch_loss/len(train_loader):.4f}, '
          f'Val Loss: {val_loss/len(val_loader):.4f}')


def visualize_predictions(model, dataset, num_images=3):
    model.eval()
    for i in range(num_images):
        image, mask = dataset[i]
        image = image.to(device).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            output = torch.sigmoid(output)
            output = output.squeeze().cpu().numpy()
            pred = (output > 0.5).astype(np.uint8)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(image.cpu().squeeze().permute(1, 2, 0))
        axs[0].set_title('Input Image')
        axs[1].imshow(mask.numpy(), cmap='gray')
        axs[1].set_title('Ground Truth')
        axs[2].imshow(pred, cmap='gray')
        axs[2].set_title('Prediction')
        for ax in axs:
            ax.axis('off')
        plt.show()


visualize_predictions(model, val_dataset)
