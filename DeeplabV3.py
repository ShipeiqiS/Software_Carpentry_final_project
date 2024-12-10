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
import datetime
import random
random.seed(233)

# 用户输入
skin_lesion_dataset_path = input("Please enter the path to the training directory: ")
test_dataset_path = input("Please enter the path to the test directory: ")
batch_size = int(input("Please enter the batch_size (e.g., 4 or 10): "))
num_epochs = int(input("Please enter the number of epochs (e.g., 10): "))
lr = float(input("Please enter the learning rate (e.g., 0.001): "))
save_address = input("Please enter the save directory for the final images: ")
pretrained_model_path = input("Please enter the path to a pre-trained model (if any), otherwise leave blank: ")

if save_address.strip() and not os.path.exists(save_address):
    os.makedirs(save_address, exist_ok=True)

# Prepare the data
image_list = [i for i in os.listdir(skin_lesion_dataset_path) if i.startswith("IMD")]
random.shuffle(image_list)

n = len(image_list)
train_list = image_list[:int(0.8 * n)]
test_list = image_list[int(0.8 * n):]

class LesionDataset(Dataset):
    def __init__(self, root_path, data_list, mode):
        self.root_path = root_path
        self.data_list = data_list
        self.mode = mode

        # Normalization transforms
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Augmentations (applied only during training)
        self.aug = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(256),
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_name = self.data_list[idx]

        # Load and normalize image
        image = PIL.Image.open(
            os.path.join(self.root_path, image_name, f"{image_name}_Dermoscopic_Image", f"{image_name}.bmp"))
        image = self.norm(image)

        # Load and binarize mask
        mask = PIL.Image.open(
            os.path.join(self.root_path, image_name, f"{image_name}_lesion", f"{image_name}_lesion.bmp"))
        mask = torch.tensor(np.array(mask)).float()
        mask[mask == 0] = 0
        mask[mask > 0] = 1

        # If training, apply joint augmentations
        if self.mode == 'train':
            data = torch.cat([image, mask.unsqueeze(0)], dim=0)
            data = self.aug(data)
            image = data[:3, :, :]
            mask = data[3:, :, :]

        return image, mask

train_dataset = LesionDataset(skin_lesion_dataset_path, train_list, 'train')
valid_dataset = LesionDataset(skin_lesion_dataset_path, test_list, "valid")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1)

# Load pretrained DeeplabV3 and modify the classifier head
model = torchvision.models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
model.classifier[-1] = torch.nn.Sequential(
    nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)),
    nn.Sigmoid()
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

# Determine whether there is a pre-trained model path, and if so, load it, otherwise train
if pretrained_model_path.strip() and os.path.exists(pretrained_model_path):
    # Load the trained model and do not train it
    print(f"Loading pretrained model from {pretrained_model_path}")
    model = torch.load(pretrained_model_path, map_location=device)
else:
    # There is no pre-trained model for training
    print("No pretrained model provided or path not found, start training...")
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

                # Pixel accuracy
                predicted = torch.zeros_like(outputs)
                predicted[outputs > 0.5] = 1.0
                predicted = predicted[0, :, :, :]
                sum_valid_correct += (predicted == labels).sum().item()
                sum_valid_pixel += outputs.shape[-1] * outputs.shape[-2]

                # Validation loss
                labels = labels.unsqueeze(0)
                sum_valid_loss += criterion(outputs, labels).item()

        sum_valid_loss = sum_valid_loss / len(valid_loader)
        sum_valid_correct = sum_valid_correct / sum_valid_pixel * 100
        print(f'Epoch: {epoch}, Validation Loss: {sum_valid_loss:.4f}, Accuracy: {sum_valid_correct:.2f}%')

        model.train()

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    model_save_path = f"DeepLabV3_{current_time}.pth"
    torch.save(model, model_save_path)
    print(f"Model saved to {model_save_path}")

# Prepare the test dataset
class NewDataLesionDataset(Dataset):
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
new_data_dataset = NewDataLesionDataset(test_dataset_path, new_data_list)
new_data_loader = DataLoader(new_data_dataset, batch_size=1)

threshold = 0.5

# We'll store results and then save them to files
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

# Save each result as a picture
for (f_name, img, pred) in results:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'{f_name} - Original')
    axes[0].axis('off')

    axes[1].imshow(pred, cmap='gray')
    axes[1].set_title(f'{f_name} - Predicted')
    axes[1].axis('off')

    # Save the file to a specified directory with f_name as the file name
    if save_address.strip():
        save_path = os.path.join(save_address, f"{f_name}_result.png")
    else:
        # If you do not specify a save address, it will be saved in the current directory
        save_path = f"{f_name}_result.png"
    plt.savefig(save_path)
    plt.close(fig)

print("All images have been saved successfully.")
