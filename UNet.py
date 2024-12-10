import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import os
import PIL
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt

random.seed(233)

# Get a list of all image folders starting with "IMD" from the dataset directory
image_list = [i for i in os.listdir("./skin_lesion_dataset") if i.startswith("IMD")]
random.shuffle(image_list)  # Shuffle the list to randomize training/validation splits

n = len(image_list)
train_list = image_list[:int(0.8 * n)]
test_list = image_list[int(0.8 * n):]


class LesionDataset(Dataset):
    """
    A custom dataset for loading skin lesion images and their corresponding masks.
    This dataset applies normalization and optional augmentations depending on the mode.
    """
    def __init__(self, root_path, data_list, mode):
        self.root_path = root_path
        self.data_list = data_list
        self.mode = mode

        # Normalization transform for the images
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Data augmentation transforms (applied only during training)
        self.aug = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(256),
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_name = self.data_list[idx]

        # Load the dermoscopic image
        image = PIL.Image.open(
            os.path.join(self.root_path, image_name, f"{image_name}_Dermoscopic_Image", f"{image_name}.bmp"))
        image = self.norm(image)

        # Load the lesion mask
        mask = PIL.Image.open(
            os.path.join(self.root_path, image_name, f"{image_name}_lesion", f"{image_name}_lesion.bmp"))
        mask = torch.tensor(np.array(mask)).float()
        mask[mask == 0] = 0
        mask[mask > 0] = 1

        # Apply augmentations if in training mode (augmentations apply to both image and mask)
        if self.mode == 'train':
            data = torch.cat([image, mask.unsqueeze(0)], dim=0)
            data = self.aug(data)
            image = data[:3, :, :]
            mask = data[3:, :, :]

        return image, mask


# Create training and validation datasets and loaders
train_dataset = LesionDataset("./skin_lesion_dataset", train_list, 'train')
valid_dataset = LesionDataset("./skin_lesion_dataset", test_list, "valid")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1)


# --------------------------
# Model Definition: UNet
# --------------------------

class DoubleConv(nn.Module):
    """
    A helper block of two convolutions each followed by batch normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downsampling block: MaxPool followed by a DoubleConv.
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upsampling block: Upsample (or ConvTranspose) followed by a DoubleConv.
    Also merges the encoder features with the upsampled features from the decoder path.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            # Use bilinear upsampling
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # Use ConvTranspose2d for upsampling
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust padding if shapes don't match exactly
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        # Concatenate encoder features (x2) with upsampled decoder features (x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """
    UNet architecture for image segmentation.
    Consists of a contracting path (down) and an expanding path (up),
    with skip connections between encoder and decoder.
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder path
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # Encoding
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoding + skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        logits = self.act(logits)
        # Return a dict to be compatible with original code structure that expected 'out'
        return {'out': logits}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=3, n_classes=1, bilinear=True)
model = model.to(device)

# Hyperparameters
batch_size = 10
num_epochs = 10
lr = 1e-3

criterion = nn.BCELoss()  # Binary cross entropy loss for segmentation
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

# Training and validation loop
model.train()
sum_train_loss = 0
sum_step = 0
for epoch in range(num_epochs):
    # Training phase
    for inputs, labels in train_loader:
        sum_step += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)['out']
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        sum_train_loss += loss.item()

    avg_train_loss = sum_train_loss / sum_step
    print(f"Epoch: {epoch}, Training Loss: {avg_train_loss}")

    # Validation phase
    model.eval()
    sum_valid_correct = 0
    sum_valid_pixel = 0
    sum_valid_loss = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)['out']

            # Calculate pixel-level accuracy
            predicted = torch.zeros_like(outputs)
            predicted[outputs > 0.5] = 1.0
            predicted = predicted[0, :, :, :]
            sum_valid_correct += (predicted == labels).sum().item()
            sum_valid_pixel += outputs.shape[-1] * outputs.shape[-2]

            # Calculate validation loss
            labels = labels.unsqueeze(0)
            sum_valid_loss += criterion(outputs, labels).item()

    sum_valid_loss = sum_valid_loss / len(valid_loader)
    sum_valid_correct = sum_valid_correct / sum_valid_pixel * 100
    print(f'Epoch: {epoch}, Validation Loss: {sum_valid_loss:.4f}, Accuracy: {sum_valid_correct:.2f}%')
    model.train()

# Save the trained model
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
torch.save(model, f"UNet_{current_time}.pth")


class NewDataLesionDataset(Dataset):
    """
    A dataset class to handle a new set of test data.
    It only applies normalization and returns the image with its folder name.
    """
    def __init__(self, root_path, data_list):
        self.root_path = root_path
        self.data_list = data_list

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        folder_name = f"Test{self.data_list[idx]:02d}"
        image_path = os.path.join(self.root_path, folder_name, f"{folder_name.lower()}.bmp")

        image = PIL.Image.open(image_path)
        image = self.norm(image)

        return image, folder_name


# Load new test data
new_data_list = list(range(1, 51))
new_data_dataset = NewDataLesionDataset("./test_dataset", new_data_list)
new_data_loader = DataLoader(new_data_dataset, batch_size=1)

# If needed, we can load a saved model:
# model = torch.load("model_xxxx-xx-xx xx-xx-xx.pth")

threshold = 0.5

# We'll store results and display them once after the loop
results = []
model.eval()
with torch.no_grad():
    for inputs, folder_names in new_data_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)['out']

        predicted = torch.zeros_like(outputs)
        predicted[outputs > threshold] = 1.0
        predicted = predicted[0, :, :, :]

        # Convert tensors to numpy for plotting later
        img_np = inputs[0, :, :, :].cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        pred_np = predicted[0, :, :].cpu().numpy()

        results.append((folder_names[0], img_np, pred_np))

# Display all results once
# Let's say we create a figure with len(results)*2 subplots in a vertical manner
fig, axes = plt.subplots(len(results), 2, figsize=(10, 5 * len(results)))
for i, (f_name, img, pred) in enumerate(results):
    axes[i, 0].imshow(img, cmap='gray')
    axes[i, 0].set_title(f'{f_name} - Original')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(pred, cmap='gray')
    axes[i, 1].set_title(f'{f_name} - Predicted')
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()
