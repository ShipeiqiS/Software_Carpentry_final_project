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

# ------------------------------------------------------------
# Prepare dataset lists
# ------------------------------------------------------------
image_list = [i for i in os.listdir("./skin_lesion_dataset") if i.startswith("IMD")]
random.shuffle(image_list)

n = len(image_list)
train_list = image_list[:int(0.8 * n)]
test_list = image_list[int(0.8 * n):]


class LesionDataset(Dataset):
    """
    A custom dataset class for loading skin lesion images and corresponding masks.
    - Normalizes the input images.
    - Optionally applies data augmentation during training.
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
# Replace DeeplabV3 model with FCN (Fully Convolutional Network)
# ------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a pre-trained FCN model (fcn_resnet50)
model = torchvision.models.segmentation.fcn_resnet50(weights="COCO_WITH_VOC_LABELS_V1")

# Modify the classifier to output a single channel and add a Sigmoid activation
model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)  # originally 21 classes, now 1
# Wrap the model forward to include a Sigmoid
old_forward = model.forward
def new_forward(x):
    out = old_forward(x)
    out['out'] = torch.sigmoid(out['out'])
    return out
model.forward = new_forward

model = model.to(device)

# Hyperparameters
batch_size = 10
num_epochs = 10
lr = 1e-3

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

# Training and validation loop
model.train()
sum_train_loss = 0
sum_step = 0
for epoch in range(num_epochs):
    # Training
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

    # Validation
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
torch.save(model, f"model_{current_time}.pth")


class NewDataLesionDataset(Dataset):
    """
    A dataset class for new test data (no masks, only images).
    Normalizes images and returns them with their folder name.
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

for inputs, folder_names in new_data_loader:
    inputs = inputs.to(device)
    outputs = model(inputs)['out']

    predicted = torch.zeros_like(outputs)
    predicted[outputs > threshold] = 1.0
    predicted = predicted[0, :, :, :]

    plt.figure()
    plt.subplot(1, 2, 1)
    image = inputs[0].cpu().permute(1, 2, 0).numpy()
    image = (image - image.min()) / (image.max() - image.min())
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    mask_img = predicted[0].cpu().numpy()
    plt.imshow(mask_img, cmap='gray')
    plt.title(folder_names[0])
    plt.show()
