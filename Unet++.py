import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import torch.nn.functional as F
import os
import PIL
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt

random.seed(233)

# ------------------------------------------------------------
# Prepare dataset lists (same as before)
# ------------------------------------------------------------
image_list = [i for i in os.listdir("./skin_lesion_dataset") if i.startswith("IMD")]
random.shuffle(image_list)

n = len(image_list)
train_list = image_list[:int(0.8 * n)]
test_list = image_list[int(0.8 * n):]


class LesionDataset(Dataset):
    """
    Same dataset class as the original code, no changes here.
    """
    def __init__(self, root_path, data_list, mode):
        self.root_path = root_path
        self.data_list = data_list
        self.mode = mode

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.aug = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(256),
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_name = self.data_list[idx]

        image = PIL.Image.open(
            os.path.join(self.root_path, image_name, f"{image_name}_Dermoscopic_Image", f"{image_name}.bmp"))
        image = self.norm(image)

        mask = PIL.Image.open(
            os.path.join(self.root_path, image_name, f"{image_name}_lesion", f"{image_name}_lesion.bmp"))
        mask = torch.tensor(np.array(mask)).float()
        mask[mask == 0] = 0
        mask[mask > 0] = 1

        if self.mode == 'train':
            data = torch.cat([image, mask.unsqueeze(0)], dim=0)
            data = self.aug(data)
            image = data[:3, :, :]
            mask = data[3:, :, :]

        return image, mask


train_dataset = LesionDataset("./skin_lesion_dataset", train_list, 'train')
valid_dataset = LesionDataset("./skin_lesion_dataset", test_list, "valid")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1)


# ------------------------------------------------------------
# Define UNet++ model
# ------------------------------------------------------------
class DoubleConv(nn.Module):
    """(Convolution => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetPP(nn.Module):
    """
    A simplified UNet++ implementation.
    UNet++ introduces dense skip connections between encoder and decoder,
    forming a "nested" U-shape to better refine feature maps.
    This is a minimal illustrative version, not necessarily fully optimized.
    """
    def __init__(self, n_channels=3, n_classes=1):
        super(UNetPP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder (same structure as UNet)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 512))

        # Nested connections (the ++ part)
        # Level 1
        self.up10 = DoubleConv(128 + 64, 64)
        self.up20 = DoubleConv(256 + 128, 128)
        self.up30 = DoubleConv(512 + 256, 256)
        self.up40 = DoubleConv(512 + 512, 512)

        # Level 2
        self.up21 = DoubleConv(128 + 64, 64)
        self.up31 = DoubleConv(256 + 128, 128)
        self.up41 = DoubleConv(512 + 256, 256)

        # Level 3
        self.up32 = DoubleConv(128 + 64, 64)
        self.up42 = DoubleConv(256 + 128, 128)

        # Level 4
        self.up43 = DoubleConv(128 + 64, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x0_0 = self.inc(x)
        x1_0 = self.down1(x0_0)
        x2_0 = self.down2(x1_0)
        x3_0 = self.down3(x2_0)
        x4_0 = self.down4(x3_0)

        # UNet++ paths
        # row 1
        x0_1 = self.up10(torch.cat([x0_0, F.interpolate(x1_0, x0_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_1 = self.up20(torch.cat([x1_0, F.interpolate(x2_0, x1_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x2_1 = self.up30(torch.cat([x2_0, F.interpolate(x3_0, x2_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x3_1 = self.up40(torch.cat([x3_0, F.interpolate(x4_0, x3_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        # row 2
        x0_2 = self.up21(torch.cat([x0_1, F.interpolate(x1_1, x0_1.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_2 = self.up31(torch.cat([x1_1, F.interpolate(x2_1, x1_1.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x2_2 = self.up41(torch.cat([x2_1, F.interpolate(x3_1, x2_1.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        # row 3
        x0_3 = self.up32(torch.cat([x0_2, F.interpolate(x1_2, x0_2.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_3 = self.up42(torch.cat([x1_2, F.interpolate(x2_2, x1_2.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        # row 4
        x0_4 = self.up43(torch.cat([x0_3, F.interpolate(x1_3, x0_3.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        logits = self.outc(x0_4)
        logits = self.act(logits)
        return {'out': logits}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNetPP(n_channels=3, n_classes=1)
model = model.to(device)

# Hyperparameters
batch_size = 10
num_epochs = 10
lr = 1e-3

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

# Training & validation
model.train()
sum_train_loss = 0
sum_step = 0
for epoch in range(num_epochs):
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
    print(f"Epoch: {epoch}, Training Loss: {avg_train_loss:.4f}")

    model.eval()
    sum_valid_correct = 0
    sum_valid_pixel = 0
    sum_valid_loss = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)['out']

            predicted = torch.zeros_like(outputs)
            predicted[outputs > 0.5] = 1.0
            predicted = predicted[0, :, :, :]
            sum_valid_correct += (predicted == labels).sum().item()
            sum_valid_pixel += outputs.shape[-1] * outputs.shape[-2]

            labels = labels.unsqueeze(0)
            sum_valid_loss += criterion(outputs, labels).item()

    sum_valid_loss = sum_valid_loss / len(valid_loader)
    sum_valid_correct = sum_valid_correct / sum_valid_pixel * 100
    print(f'Epoch: {epoch}, Validation Loss: {sum_valid_loss:.4f}, Accuracy: {sum_valid_correct:.2f}%')
    model.train()

# Save the model
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
torch.save(model, f"UNet++_{current_time}.pth")


class NewDataLesionDataset(Dataset):
    """
    Same as previous code, used for inference on new test images.
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

new_data_list = list(range(1, 51))
new_data_dataset = NewDataLesionDataset("./test_dataset", new_data_list)
new_data_loader = DataLoader(new_data_dataset, batch_size=1)

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
