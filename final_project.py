import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import os
import PIL
from PIL import Image, ImageTk
import numpy as np
import random
import datetime
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

random.seed(233)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LesionDataset(Dataset):
    """
    A custom dataset class for loading training/validation images and masks.
    """
    def __init__(self, root_path, data_list, mode):
        self.root_path = root_path
        self.data_list = data_list
        self.mode = mode

        # Normalization transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Data augmentation
        self.aug = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(256),
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_name = self.data_list[idx]

        # Load and normalize the image
        image = PIL.Image.open(
            os.path.join(self.root_path, image_name, f"{image_name}_Dermoscopic_Image", f"{image_name}.bmp"))
        image = self.norm(image)

        # Load and binarize the mask
        mask = PIL.Image.open(
            os.path.join(self.root_path, image_name, f"{image_name}_lesion", f"{image_name}_lesion.bmp"))
        mask = torch.tensor(np.array(mask)).float()
        mask[mask == 0] = 0
        mask[mask > 0] = 1

        # If in training mode, apply augmentations to both image and mask
        if self.mode == 'train':
            data = torch.cat([image, mask.unsqueeze(0)], dim=0)
            data = self.aug(data)
            image = data[:3, :, :]
            mask = data[3:, :, :]

        return image, mask

class NewDataLesionDataset(Dataset):
    """
    A dataset class for new test data (no masks). Only applies normalization.
    """
    def __init__(self, root_path, data_list):
        self.root_path = root_path
        self.data_list = data_list

        # Normalization transform
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

def get_deeplabv3():
    """
    Get a pre-trained DeeplabV3 model and modify the classifier head
    to output a single channel for binary segmentation with Sigmoid.
    """
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
    model.classifier[-1] = nn.Sequential(
        nn.Conv2d(256, 1, kernel_size=1),
        nn.Sigmoid()
    )
    return model

def get_fcn():
    """
    Get a pre-trained FCN model, modify the last layer for binary segmentation,
    and add a Sigmoid activation.
    """
    from torchvision.models.segmentation import fcn_resnet50
    model = fcn_resnet50(weights="COCO_WITH_VOC_LABELS_V1")
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
    old_forward = model.forward
    def new_forward(x):
        out = old_forward(x)
        out['out'] = torch.sigmoid(out['out'])
        return out
    model.forward = new_forward
    return model

class DoubleConv(nn.Module):
    """
    A double convolution block
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
    Downsampling block
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
    Upsampling block
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: decoder input, x2: encoder feature map
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """
    A UNet model for segmentation, output 1 channel with Sigmoid.
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
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
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.act = nn.Sigmoid()

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
        logits = self.act(logits)
        return {'out': logits}

class UNetPP(nn.Module):
    """
    A simplified UNet++ model structure for segmentation (1-channel output + Sigmoid).
    """
    def __init__(self, n_channels=3, n_classes=1):
        super(UNetPP, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 512))

        self.up10 = DoubleConv(64+128, 64)
        self.up20 = DoubleConv(128+256, 128)
        self.up30 = DoubleConv(256+512, 256)
        self.up40 = DoubleConv(512+512, 512)

        self.up21 = DoubleConv(64+128, 64)
        self.up31 = DoubleConv(128+256, 128)
        self.up41 = DoubleConv(256+512, 256)

        self.up32 = DoubleConv(64+128, 64)
        self.up42 = DoubleConv(128+256, 128)

        self.up43 = DoubleConv(64+128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x0_0 = self.inc(x)
        x1_0 = self.down1(x0_0)
        x2_0 = self.down2(x1_0)
        x3_0 = self.down3(x2_0)
        x4_0 = self.down4(x3_0)

        x0_1 = self.up10(torch.cat([x0_0, F.interpolate(x1_0, x0_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_1 = self.up20(torch.cat([x1_0, F.interpolate(x2_0, x1_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x2_1 = self.up30(torch.cat([x2_0, F.interpolate(x3_0, x2_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x3_1 = self.up40(torch.cat([x3_0, F.interpolate(x4_0, x3_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        x0_2 = self.up21(torch.cat([x0_1, F.interpolate(x1_1, x0_1.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_2 = self.up31(torch.cat([x1_1, F.interpolate(x2_1, x1_1.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x2_2 = self.up41(torch.cat([x2_1, F.interpolate(x3_1, x2_1.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        x0_3 = self.up32(torch.cat([x0_2, F.interpolate(x1_2, x0_2.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_3 = self.up42(torch.cat([x1_2, F.interpolate(x2_2, x1_2.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        x0_4 = self.up43(torch.cat([x0_3, F.interpolate(x1_3, x0_3.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        logits = self.outc(x0_4)
        logits = self.act(logits)
        return {'out': logits}

class MedicalImageSegmentationGUI(tk.Tk):
    """
    A GUI application for medical image segmentation using various models (DeepLabV3, FCN, UNet, UNet++).
    Allows loading datasets, specifying parameters, optionally loading pretrained weights,
    training the model, and then displaying the segmentation results of test images.
    """
    def __init__(self):
        super().__init__()
        self.title("Medical Image Segmentation Expert")

        # Title
        title_label = tk.Label(self, text="Medical Image Segmentation Expert", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=10)

        # Parameter input frame
        params_frame = tk.Frame(self)
        params_frame.pack(pady=5)

        tk.Label(params_frame, text="Batch Size:").grid(row=0, column=0, padx=5, pady=5)
        self.batch_size_entry = tk.Entry(params_frame)
        self.batch_size_entry.grid(row=0, column=1, padx=5, pady=5)
        self.batch_size_entry.insert(0, "5")

        tk.Label(params_frame, text="Total Epochs:").grid(row=0, column=2, padx=5, pady=5)
        self.epochs_entry = tk.Entry(params_frame)
        self.epochs_entry.grid(row=0, column=3, padx=5, pady=5)
        self.epochs_entry.insert(0, "10")

        tk.Label(params_frame, text="Learning Rate:").grid(row=0, column=4, padx=5, pady=5)
        self.lr_entry = tk.Entry(params_frame)
        self.lr_entry.grid(row=0, column=5, padx=5, pady=5)
        self.lr_entry.insert(0, "0.001")

        # Model selection frame
        model_frame = tk.Frame(self)
        model_frame.pack(pady=5)

        tk.Label(model_frame, text="Select Model:").grid(row=0, column=0, padx=5, pady=5)
        self.model_var = tk.StringVar(value="DeepLabV3")
        models = ["DeepLabV3", "FCN", "UNet", "UNet++"]
        for i, m in enumerate(models):
            tk.Radiobutton(model_frame, text=m, variable=self.model_var, value=m).grid(row=0, column=i+1, padx=5)

        # Path input frame
        path_frame = tk.Frame(self)
        path_frame.pack(pady=5)

        tk.Label(path_frame, text="Train Dataset:").grid(row=0, column=0, padx=5, pady=5)
        self.train_path_entry = tk.Entry(path_frame, width=40)
        self.train_path_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(path_frame, text="Test Dataset:").grid(row=1, column=0, padx=5, pady=5)
        self.test_path_entry = tk.Entry(path_frame, width=40)
        self.test_path_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(path_frame, text="Save Address:").grid(row=2, column=0, padx=5, pady=5)
        self.save_path_entry = tk.Entry(path_frame, width=40)
        self.save_path_entry.grid(row=2, column=1, padx=5, pady=5)

        tk.Label(path_frame, text="Pre-trained Model:").grid(row=3, column=0, padx=5, pady=5)
        self.pretrained_entry = tk.Entry(path_frame, width=40)
        self.pretrained_entry.grid(row=3, column=1, padx=5, pady=5)

        # RUN button
        run_button = tk.Button(self, text="RUN", command=self.run_segmentation)
        run_button.pack(pady=10)

        tk.Label(self, text="Logs:").pack(pady=(10,0))
        self.logs_text = ScrolledText(self, width=80, height=10)
        self.logs_text.pack(pady=5)

        # Bottom frame for training curve and test results
        bottom_frame = tk.Frame(self)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Left frame: Training Curve
        left_frame = tk.Frame(bottom_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(left_frame, text="Training Curve").pack(pady=5)

        self.fig = Figure(figsize=(1,6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Accuracy Curve")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Accuracy (%)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right frame: Test Results
        right_frame = tk.Frame(bottom_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(right_frame, text="Test Results").pack(pady=5)

        self.results_canvas = tk.Canvas(right_frame, width=400, height=400)
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(right_frame, orient="vertical", command=self.results_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.results_frame = tk.Frame(self.results_canvas)
        self.results_canvas.create_window((0, 0), window=self.results_frame, anchor='nw')

        self.results_frame.bind("<Configure>",
                                lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all")))

    def log(self, msg):
        """
        Utility function to print logs to the ScrolledText widget.
        """
        self.logs_text.insert(tk.END, msg + "\n")
        self.logs_text.see(tk.END)
        self.update()

    def run_segmentation(self):
        """
        Main function to run the segmentation process:
        1) Get parameters and paths.
        2) Build model according to user selection.
        3) Load pretrained weights if provided.
        4) Train the model if no pretrained weights are given.
        5) Validate the model and plot accuracy curve.
        6) Run inference on new test data and display all result images stitched vertically.
        """
        batch_size = int(self.batch_size_entry.get())
        num_epochs = int(self.epochs_entry.get())
        lr = float(self.lr_entry.get())
        model_choice = self.model_var.get()
        skin_lesion_dataset_path = self.train_path_entry.get().strip()
        test_dataset_path = self.test_path_entry.get().strip()
        save_address = self.save_path_entry.get().strip()
        pretrained_model_path = self.pretrained_entry.get().strip()

        # Check for valid paths
        if not skin_lesion_dataset_path or not os.path.exists(skin_lesion_dataset_path):
            messagebox.showerror("Error", "Invalid Train Dataset path.")
            return
        if not test_dataset_path or not os.path.exists(test_dataset_path):
            messagebox.showerror("Error", "Invalid Test Dataset path.")
            return
        if save_address and not os.path.exists(save_address):
            os.makedirs(save_address, exist_ok=True)

        self.log("Starting Segmentation Process...")
        self.log(f"Model: {model_choice}")
        self.log(f"Batch Size: {batch_size}, Epochs: {num_epochs}, LR: {lr}")
        self.log(f"Train Path: {skin_lesion_dataset_path}")
        self.log(f"Test Path: {test_dataset_path}")
        self.log(f"Save Path: {save_address}")
        if pretrained_model_path:
            self.log(f"Using Pre-trained Model (weights): {pretrained_model_path}")
        else:
            self.log("No Pre-trained weights provided, training from scratch.")

        # Build model according to selection
        if model_choice == 'DeepLabV3':
            model = get_deeplabv3()
            model_name = "DeepLabV3"
        elif model_choice == 'FCN':
            model = get_fcn()
            model_name = "FCN"
        elif model_choice == 'UNet':
            model = UNet(n_channels=3, n_classes=1, bilinear=True)
            model_name = "UNet"
        elif model_choice == 'UNet++':
            model = UNetPP(n_channels=3, n_classes=1)
            model_name = "UNet++"
        else:
            self.log("Invalid model choice, defaulting to DeepLabV3.")
            model = get_deeplabv3()
            model_name = "DeepLabV3"

        model = model.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

        # Prepare train/valid split
        image_list = [i for i in os.listdir(skin_lesion_dataset_path) if i.startswith("IMD")]
        random.shuffle(image_list)
        n = len(image_list)
        train_list = image_list[:int(0.8 * n)]
        test_list = image_list[int(0.8 * n):]

        train_dataset = LesionDataset(skin_lesion_dataset_path, train_list, 'train')
        valid_dataset = LesionDataset(skin_lesion_dataset_path, test_list, "valid")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=1)

        # Load pretrained weights if provided
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            self.log(f"Loading weights from {pretrained_model_path}")
            state_dict = torch.load(pretrained_model_path, map_location=device)
            model.load_state_dict(state_dict)
            self.log("Weights loaded successfully. Skipping training ...")
            trained = True
            accuracy_list = []
        else:
            self.log("No valid pretrained weights provided, starting training...")
            model.train()
            sum_train_loss = 0
            sum_step = 0
            accuracy_list = []

            # Train the model
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

                # Validation phase
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
                val_acc = sum_valid_correct / sum_valid_pixel * 100
                accuracy_list.append(val_acc)

                self.log(f'Epoch: {epoch}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {sum_valid_loss:.4f}, Accuracy: {val_acc:.2f}%')

                model.train()

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            model_save_path = f"{model_name}_{current_time}.pth"
            torch.save(model.state_dict(), model_save_path)
            self.log(f"Model weights saved to {model_save_path}")
            trained = True

        # Update accuracy curve
        if accuracy_list:
            self.ax.clear()
            self.ax.set_title("Accuracy Curve")
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel("Accuracy (%)")
            self.ax.plot(range(len(accuracy_list)), accuracy_list, marker='o')
            self.canvas.draw()

        # Inference on new data
        new_data_list = list(range(1, 51))
        new_data_dataset = NewDataLesionDataset(test_dataset_path, new_data_list)
        new_data_loader = DataLoader(new_data_dataset, batch_size=1)

        threshold = 0.5
        model.eval()
        all_image_paths = []

        with torch.no_grad():
            for inputs, folder_names in new_data_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)['out']

                predicted = torch.zeros_like(outputs)
                predicted[outputs > threshold] = 1.0
                predicted = predicted[0, :, :, :]

                img_np = inputs[0, :, :, :].cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                pred_np = predicted[0, :, :].cpu().numpy()

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(img_np, cmap='gray')
                axes[0].set_title(f'{folder_names[0]} - Original')
                axes[0].axis('off')

                axes[1].imshow(pred_np, cmap='gray')
                axes[1].set_title(f'{folder_names[0]} - Predicted')
                axes[1].axis('off')

                if save_address.strip():
                    save_path = os.path.join(save_address, f"{folder_names[0]}_result.png")
                else:
                    save_path = f"{folder_names[0]}_result.png"

                plt.savefig(save_path)
                plt.close(fig)

                all_image_paths.append(save_path)

        self.log("All images have been saved successfully.")

        # Combine all test result images vertically and display in GUI
        if len(all_image_paths) > 0:
            images = [Image.open(p) for p in all_image_paths]
            widths, heights = zip(*(im.size for im in images))
            max_width = max(widths)
            total_height = sum(heights)

            combined_img = Image.new('RGB', (max_width, total_height), (255, 255, 255))
            y_offset = 0
            for im in images:
                x_offset = (max_width - im.width) // 2
                combined_img.paste(im, (x_offset, y_offset))
                y_offset += im.size[1]

            self.result_img_tk = ImageTk.PhotoImage(combined_img)
            lbl = tk.Label(self.results_frame, image=self.result_img_tk)
            lbl.pack()

            self.results_frame.update_idletasks()
            self.results_canvas.config(scrollregion=self.results_canvas.bbox("all"))
        else:
            self.log("No test result image found to display.")

if __name__ == "__main__":
    app = MedicalImageSegmentationGUI()
    app.mainloop()
