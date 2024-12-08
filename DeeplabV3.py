import torch
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import os, sys, json, tqdm
import PIL
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torch import nn, optim
from sklearn.model_selection import KFold
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights
import matplotlib.pyplot as plt
import random

random.seed(233)

# Create a shuffled list of image folder names
image_list = [i for i in os.listdir("./skin_lesion_dataset") if i.startswith("IMD")]
random.shuffle(image_list)

# Split data into train (80%) and test (20%)
n = len(image_list)
train_list = image_list[:int(0.8 * n)]
test_list = image_list[int(0.8 * n):]


class LesionDataset(Dataset):
    def __init__(self, root_path, data_list, mode):
        self.root_path = root_path
        self.data_list = data_list
        self.mode = mode

        # Normalization transform for input images
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Data augmentation transforms applied only when training
        self.aug = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(256),
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Load the image
        image_name = self.data_list[idx]
        image = PIL.Image.open(
            os.path.join(self.root_path, image_name, f"{image_name}_Dermoscopic_Image", f"{image_name}.bmp"))
        image = self.norm(image)

        # Load the corresponding mask and binarize it
        mask = PIL.Image.open(
            os.path.join(self.root_path, image_name, f"{image_name}_lesion", f"{image_name}_lesion.bmp"))
        mask = torch.tensor(np.array(mask)).float()
        mask[mask == 0] = 0
        mask[mask > 0] = 1

        # Apply augmentation only in training mode
        if self.mode == 'train':
            # Concatenate image and mask to jointly apply transformations
            data = torch.cat([image, mask.unsqueeze(0)], dim=0)
            data = self.aug(data)
            # Split them back after augmentation
            image = data[:3, :, :]
            mask = data[3:, :, :]

        return image, mask


# Create training and validation datasets and dataloaders
train_dataset = LesionDataset("./skin_lesion_dataset", train_list, 'train')
valid_dataset = LesionDataset("./skin_lesion_dataset", test_list, "valid")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1)

# Load a pre-trained DeepLabV3 model (ResNet-50 backbone)
# and modify the classifier layer to output a single channel (binary segmentation)
model = torchvision.models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
model.classifier[-1] = torch.nn.Sequential(
    nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)),
    nn.Sigmoid()
)

# Set device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Set hyperparameters
batch_size = 10
num_epochs = 10
record_gap = 20
lr = 1e-3

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

# Training and validation
model.train()
sum_train_loss = 0
sum_step = 0
for epoch in range(num_epochs):
    # Training loop
    for inputs, labels in train_loader:
        sum_step += 1
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)['out']
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate training loss
        sum_train_loss += loss.item()

    # Compute average training loss for this epoch
    avg_train_loss = sum_train_loss / sum_step
    print(f"Epoch: {epoch}, Training Loss: {avg_train_loss}")

    # Validation loop
    model.eval()
    sum_valid_correct = 0
    sum_valid_pixel = 0
    sum_valid_loss = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)['out']

            # Compute pixel-wise accuracy
            predicted = torch.zeros_like(outputs)
            predicted[outputs > 0.5] = 1.0
            predicted = predicted[0, :, :, :]
            sum_valid_correct += (predicted == labels).sum().item()
            sum_valid_pixel += outputs.shape[-1] * outputs.shape[-2]

            # Compute validation loss
            labels = labels.unsqueeze(0)
            sum_valid_loss += criterion(outputs, labels).item()

    # Average validation loss and pixel accuracy
    sum_valid_loss = sum_valid_loss / len(valid_loader)
    sum_valid_correct = sum_valid_correct / sum_valid_pixel * 100
    print(f'Epoch: {epoch}, Validation Loss: {sum_valid_loss:.4f}, Accuracy: {sum_valid_correct:.2f}%')

    # Switch back to training mode for next epoch
    model.train()

# Save the trained model
import datetime

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
torch.save(model, f"model_{current_time}.pth")


# A new dataset class for testing on new data
class NewDataLesionDataset(Dataset):
    def __init__(self, root_path, data_list):
        self.root_path = root_path
        self.data_list = data_list

        # Normalization transform only
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Construct image path and load the image
        folder_name = f"Test{self.data_list[idx]:02d}"
        image_path = os.path.join(self.root_path, folder_name, f"{folder_name.lower()}.bmp")

        image = PIL.Image.open(image_path)
        image = self.norm(image)

        return image, folder_name


# Create a new dataset and dataloader for inference on new images
new_data_list = list(range(1, 51))
new_data_dataset = NewDataLesionDataset("./test_dataset", new_data_list)
new_data_loader = DataLoader(new_data_dataset, batch_size=1)

# You can load a previously saved model if needed:
# model = torch.load("model_XXXX-XX-XX XX-XX-XX.pth")

threshold = 0.5

# Run inference on the new data and visualize results
for inputs, folder_names in new_data_loader:
    inputs = inputs.to(device)
    outputs = model(inputs)['out']

    # Binarize the output mask using a threshold
    predicted = torch.zeros_like(outputs)
    predicted[outputs > threshold] = 1.0
    predicted = predicted[0, :, :, :]

    # Plot the original image and the predicted mask
    plt.figure()
    plt.subplot(1, 2, 1)
    image = inputs[0, :, :, :].transpose(0, 2).transpose(0, 1).cpu().numpy()
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    predicted = predicted[0, :, :].cpu().numpy()
    plt.imshow(predicted, cmap='gray')
    plt.title(folder_names[0])
    plt.show()
