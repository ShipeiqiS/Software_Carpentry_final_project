import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# U-Net Model
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = self.double_conv(n_channels, 64)
        self.down1 = self.down(64, 128)
        self.down2 = self.down(128, 256)
        self.down3 = self.down(256, 512)
        self.down4 = self.down(512, 512)
        self.up1 = self.up(1024, 256)
        self.up2 = self.up(512, 128)
        self.up3 = self.up(256, 64)
        self.up4 = self.up(128, 64)
        self.outc = self.out_conv(64, n_classes)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self.double_conv(in_channels, out_channels)
        )

    def up(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def out_conv(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(torch.cat([x4, F.interpolate(x5, x4.size()[2:], mode='bilinear', align_corners=True)], dim=1))
        x = self.up2(torch.cat([x3, F.interpolate(x, x3.size()[2:], mode='bilinear', align_corners=True)], dim=1))
        x = self.up3(torch.cat([x2, F.interpolate(x, x2.size()[2:], mode='bilinear', align_corners=True)], dim=1))
        x = self.up4(torch.cat([x1, F.interpolate(x, x1.size()[2:], mode='bilinear', align_corners=True)], dim=1))
        logits = self.outc(x)
        logits = F.interpolate(logits, size=(256, 256), mode='bilinear', align_corners=True)
        return logits


# U-Net++ Model
class UNetPlusPlus(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetPlusPlus, self).__init__()
        self.inc = self.double_conv(n_channels, 64)
        self.down1 = self.down(64, 128)
        self.down2 = self.down(128, 256)
        self.down3 = self.down(256, 512)
        self.down4 = self.down(512, 512)
        self.up1 = self.up(1024, 256)
        self.up2 = self.up(512, 128)
        self.up3 = self.up(256, 64)
        self.up4 = self.up(128, 64)
        self.outc = self.out_conv(64, n_classes)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self.double_conv(in_channels, out_channels)
        )

    def up(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def out_conv(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(torch.cat([x4, F.interpolate(x5, x4.size()[2:], mode='bilinear', align_corners=True)], dim=1))
        x = self.up2(torch.cat([x3, F.interpolate(x, x3.size()[2:], mode='bilinear', align_corners=True)], dim=1))
        x = self.up3(torch.cat([x2, F.interpolate(x, x2.size()[2:], mode='bilinear', align_corners=True)], dim=1))
        x = self.up4(torch.cat([x1, F.interpolate(x, x1.size()[2:], mode='bilinear', align_corners=True)], dim=1))
        logits = self.outc(x)
        logits = F.interpolate(logits, size=(256, 256), mode='bilinear', align_corners=True)
        return logits


# DeepLabV3 Model
def replace_bn_with_gn(model):
    """
    Recursively replace all BatchNorm layers in a model with GroupNorm.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Replace with GroupNorm (use 32 groups or adjust as needed)
            setattr(model, name, nn.GroupNorm(32, module.num_features))
        else:
            # Recursively apply to child modules
            replace_bn_with_gn(module)


class DeepLabV3(nn.Module):
    def __init__(self, n_classes):
        super(DeepLabV3, self).__init__()
        from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

        # Modify the first convolution layer to accept single-channel input
        self.model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace all BatchNorm layers with GroupNorm
        replace_bn_with_gn(self.model)

        # Adjust classifier to match the number of output classes
        self.model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.model(x)['out']
        # Ensure spatial dimensions are larger than 1x1
        if x.size(2) < 2 or x.size(3) < 2:
            x = F.interpolate(x, size=(2, 2), mode='bilinear', align_corners=True)
        return x


class FCN(nn.Module):
    def __init__(self, n_classes):
        super(FCN, self).__init__()
        from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
        self.model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
        # Modify the first convolution layer to accept single-channel input
        self.model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Adjust classifier to match the number of output classes
        self.model.classifier[4] = nn.Conv2d(512, n_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']



class MedicalImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Image Segmentation Expert")

        # Set the window size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.9)
        self.root.geometry(f"{window_width}x{window_height}")

        # Current selected model
        self.selected_model = None
        self.models = {
            "U-Net": UNet(n_channels=1, n_classes=2),
            "DeepLabV3": DeepLabV3(n_classes=2),
            "FCN": FCN(n_classes=2),
            "UNet++": UNetPlusPlus(n_channels=1, n_classes=2)
        }

        self.train_progress = []
        self.epoch_numbers = []

        # Layout setup
        self.setup_layout(window_width, window_height)

    def setup_layout(self, window_width, window_height):
        title_label = tk.Label(self.root, text="Medical Image Segmentation Expert", font=("Arial", 18, "bold"))
        title_label.pack(pady=10)

        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        self.add_control_buttons(control_frame)
        self.add_save_address(control_frame)
        self.add_model_selection()
        self.add_dataset_inputs()
        self.add_log_area()
        self.add_result_area()

    def add_control_buttons(self, frame):
        tk.Label(frame, text="Batch Size:", font=("Arial", 12)).grid(row=0, column=0, padx=5)
        tk.Button(frame, text="-", command=self.decrease_batch_size, font=("Arial", 12)).grid(row=0, column=1, padx=5)
        self.batch_size_label = tk.Label(frame, text="5", width=5, relief=tk.SUNKEN, font=("Arial", 12))
        self.batch_size_label.grid(row=0, column=2, padx=5)
        tk.Button(frame, text="+", command=self.increase_batch_size, font=("Arial", 12)).grid(row=0, column=3, padx=5)

        tk.Label(frame, text="Total Epochs:", font=("Arial", 12)).grid(row=0, column=4, padx=5)
        tk.Button(frame, text="-", command=self.decrease_epochs, font=("Arial", 12)).grid(row=0, column=5, padx=5)
        self.epochs_label = tk.Label(frame, text="100", width=5, relief=tk.SUNKEN, font=("Arial", 12))
        self.epochs_label.grid(row=0, column=6, padx=5)
        tk.Button(frame, text="+", command=self.increase_epochs, font=("Arial", 12)).grid(row=0, column=7, padx=5)

        tk.Label(frame, text="Learning Rate:", font=("Arial", 12)).grid(row=0, column=8, padx=5)
        tk.Button(frame, text="-", command=self.decrease_learning_rate, font=("Arial", 12)).grid(row=0, column=9, padx=5)
        self.learning_rate_label = tk.Label(frame, text="0.02", width=5, relief=tk.SUNKEN, font=("Arial", 12))
        self.learning_rate_label.grid(row=0, column=10, padx=5)
        tk.Button(frame, text="+", command=self.increase_learning_rate, font=("Arial", 12)).grid(row=0, column=11, padx=5)

    def decrease_batch_size(self):
        """Decrease batch size by 1, ensuring it stays above 1."""
        current_size = int(self.batch_size_label["text"])
        if current_size > 1:
            self.batch_size_label["text"] = str(current_size - 1)

    def increase_batch_size(self):
        """Increase batch size by 1."""
        current_size = int(self.batch_size_label["text"])
        self.batch_size_label["text"] = str(current_size + 1)

    def decrease_epochs(self):
        """Decrease total epochs by 1, ensuring it stays above 1."""
        current_epochs = int(self.epochs_label["text"])
        if current_epochs > 1:
            self.epochs_label["text"] = str(current_epochs - 1)

    def increase_epochs(self):
        """Increase total epochs by 1."""
        current_epochs = int(self.epochs_label["text"])
        self.epochs_label["text"] = str(current_epochs + 1)

    def decrease_learning_rate(self):
        """Decrease learning rate, ensuring it stays above a minimal value."""
        current_rate = float(self.learning_rate_label["text"])
        if current_rate > 0.01:
            self.learning_rate_label["text"] = f"{current_rate - 0.01:.2f}"

    def increase_learning_rate(self):
        """Increase learning rate."""
        current_rate = float(self.learning_rate_label["text"])
        self.learning_rate_label["text"] = f"{current_rate + 0.01:.2f}"

    def add_save_address(self, frame):
        tk.Label(frame, text="Save Address:", font=("Arial", 12)).grid(row=1, column=0, padx=5)
        self.save_address_entry = tk.Entry(frame, width=50)
        self.save_address_entry.grid(row=1, column=1, columnspan=10, padx=5, pady=10)

        run_button = tk.Button(self.root, text="Run", font=("Arial", 14, "bold"), bg="green", fg="white",
                               command=self.start_training)
        run_button.pack(pady=10)

    def add_result_area(self):
        """
        Add areas for displaying training curves and test results.
        """
        # Create a main frame for results
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Training curve area
        train_curve_frame = tk.Frame(main_frame, relief=tk.SUNKEN, bd=2, bg="white")
        train_curve_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        tk.Label(train_curve_frame, text="Epoch Training Curve", font=("Arial", 14, "bold"), bg="white").pack(pady=10)

        # Matplotlib Figure for Training Curve
        figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax = figure.add_subplot(111)
        self.ax.grid(True)
        self.ax.set_title("Training Curve")
        self.canvas = FigureCanvasTkAgg(figure, train_curve_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Test results display area
        results_frame = tk.Frame(main_frame, relief=tk.SUNKEN, bd=2, bg="white")
        results_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        tk.Label(results_frame, text="Sample Test Results", font=("Arial", 14, "bold"), fg="blue", bg="white").pack(
            pady=10)

        # Create grid for result boxes
        self.result_boxes_frame = tk.Frame(results_frame, bg="white")
        self.result_boxes_frame.pack(fill=tk.BOTH, expand=True)

        self.result_boxes = []  # Store result display boxes
        for i in range(2):  # 2 rows
            for j in range(3):  # 3 columns
                result_box = tk.Frame(self.result_boxes_frame, width=320, height=220, relief=tk.SUNKEN, bd=2,
                                      bg="white")
                result_box.grid(row=i, column=j, padx=20, pady=20)
                self.result_boxes.append(result_box)

    def add_dataset_inputs(self):
        """Add dataset input fields for train and test datasets and labels."""
        dataset_frame = tk.Frame(self.root)
        dataset_frame.pack(pady=10)

        tk.Label(dataset_frame, text="Train Dataset:", font=("Arial", 12)).grid(row=0, column=0, padx=5, sticky="e")
        self.train_dataset_entry = tk.Entry(dataset_frame, width=40)
        self.train_dataset_entry.grid(row=0, column=1, padx=5)

        tk.Label(dataset_frame, text="Train Label:", font=("Arial", 12)).grid(row=1, column=0, padx=5, sticky="e")
        self.train_label_entry = tk.Entry(dataset_frame, width=40)
        self.train_label_entry.grid(row=1, column=1, padx=5)

        tk.Label(dataset_frame, text="Test Dataset:", font=("Arial", 12)).grid(row=2, column=0, padx=5, sticky="e")
        self.test_dataset_entry = tk.Entry(dataset_frame, width=40)
        self.test_dataset_entry.grid(row=2, column=1, padx=5)

        tk.Label(dataset_frame, text="Test Label:", font=("Arial", 12)).grid(row=3, column=0, padx=5, sticky="e")
        self.test_label_entry = tk.Entry(dataset_frame, width=40)
        self.test_label_entry.grid(row=3, column=1, padx=5)

    def add_log_area(self):
        """
        Create a log area to display messages during the training and testing process.
        """
        log_frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=2)
        log_frame.pack(fill=tk.BOTH, expand=False, padx=20, pady=10)
        tk.Label(log_frame, text="Logs", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)

        # Create a scrollable text area for logs
        self.log_area = scrolledtext.ScrolledText(log_frame, width=100, height=10, state="disabled")
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def log_message(self, message):
        """
        Append a message to the log area in the GUI.
        """
        self.log_area.configure(state="normal")  # Enable editing in the log area
        self.log_area.insert(tk.END, message + "\n")  # Append the message
        self.log_area.see(tk.END)  # Scroll to the latest message
        self.log_area.configure(state="disabled")  # Disable editing to prevent user modification

    def add_model_selection(self):
        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=10)
        tk.Label(model_frame, text="Select Model:", font=("Arial", 12)).grid(row=0, column=0, padx=5)

        models = list(self.models.keys())
        self.model_buttons = []
        for i, model_name in enumerate(models):
            btn = tk.Button(
                model_frame, text=model_name, font=("Arial", 12),
                command=lambda idx=i: self.select_model(idx)
            )
            btn.grid(row=0, column=i + 1, padx=5)
            self.model_buttons.append(btn)

    def select_model(self, model_index):
        for i, btn in enumerate(self.model_buttons):
            if i == model_index:
                btn.config(bg="green", fg="white")
                self.selected_model = list(self.models.keys())[i]
            else:
                btn.config(bg="SystemButtonFace", fg="black")

    def validate_paths(self):
        paths = [
            ("Train Dataset", self.train_dataset_entry.get()),
            ("Train Label", self.train_label_entry.get()),
            ("Test Dataset", self.test_dataset_entry.get()),
            ("Test Label", self.test_label_entry.get()),
            ("Save Address", self.save_address_entry.get())
        ]
        for label, path in paths:
            if not os.path.isdir(path):
                messagebox.showerror("Error", f"{label} path is invalid: {path}")
                return False

        # Additional validation: Ensure train and test sets have matching numbers of files
        train_data_files = os.listdir(self.train_dataset_entry.get())
        train_label_files = os.listdir(self.train_label_entry.get())
        if len(train_data_files) != len(train_label_files):
            messagebox.showerror("Error", "Mismatch between train images and labels.")
            return False

        test_data_files = os.listdir(self.test_dataset_entry.get())
        test_label_files = os.listdir(self.test_label_entry.get())
        if len(test_data_files) != len(test_label_files):
            messagebox.showerror("Error", "Mismatch between test images and labels.")
            return False

        return True


    def start_training(self):
        if not self.validate_paths():
            return

        if not self.selected_model:
            messagebox.showerror("Error", "Please select a model before running.")
            return

        batch_size = int(self.batch_size_label["text"])
        epochs = int(self.epochs_label["text"])
        learning_rate = float(self.learning_rate_label["text"])

        self.log_message(f"Starting training with {self.selected_model}...")
        self.log_message(f"Batch Size: {batch_size}, Epochs: {epochs}, Learning Rate: {learning_rate}")

        # Use threading to prevent GUI freezing
        threading.Thread(target=self.run_training_process, args=(batch_size, epochs, learning_rate)).start()

    def load_data(self, train_data_path, train_label_path, test_data_path, test_label_path):
        """
        Load and preprocess the train and test datasets.

        Args:
            train_data_path (str): Path to the training images directory.
            train_label_path (str): Path to the training labels directory.
            test_data_path (str): Path to the testing images directory.
            test_label_path (str): Path to the testing labels directory.

        Returns:
            tuple: DataLoader objects for training and testing datasets.
        """

        class CustomDataset(Dataset):
            def __init__(self, image_dir, label_dir, transform=None, target_size=(256, 256)):
                self.image_dir = image_dir
                self.label_dir = label_dir
                self.image_files = os.listdir(image_dir)
                self.label_files = os.listdir(label_dir)
                self.transform = transform
                self.target_size = target_size

            def __len__(self):
                return len(self.image_files)

            def __getitem__(self, idx):
                image_path = os.path.join(self.image_dir, self.image_files[idx])
                label_path = os.path.join(self.label_dir, self.label_files[idx])
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

                # Resize images and labels to the target size
                image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, self.target_size, interpolation=cv2.INTER_NEAREST)

                # Convert to NumPy arrays
                image = np.expand_dims(image, axis=2)  # Add a channel dimension
                label = np.expand_dims(label, axis=2)  # Add a channel dimension

                # Convert to PIL.Image
                image = Image.fromarray(image.squeeze(), mode="L")
                label = Image.fromarray(label.squeeze(), mode="L")

                # Apply transformations
                if self.transform:
                    image = self.transform(image)
                    label = transforms.ToTensor()(label).squeeze(0)  # Convert label to [H, W]

                return image, label

        transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Resize images to larger size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        train_dataset = CustomDataset(train_data_path, train_label_path, transform)
        test_dataset = CustomDataset(test_data_path, test_label_path, transform)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        return train_loader, test_loader

    def run_training_process(self, batch_size, epochs, learning_rate):
        try:
            train_data_path = self.train_dataset_entry.get()
            train_label_path = self.train_label_entry.get()
            test_data_path = self.test_dataset_entry.get()
            test_label_path = self.test_label_entry.get()
            save_address = self.save_address_entry.get()

            # Load datasets
            # Inside run_training_process
            train_loader, test_loader = self.load_data(
                train_data_path, train_label_path, test_data_path, test_label_path
            )

            # Get selected model
            model = self.models[self.selected_model]
            device = torch.device("cpu")  # Explicitly set to use CPU
            model = model.to(device)  # Move the model to CPU

            # Define optimizer and loss
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = torch.nn.CrossEntropyLoss()

            # Training loop
            self.log_message("Training started...")
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)  # Move data to the selected device
                    labels = labels.squeeze(1)  # Remove the channel dimension if it exists
                    outputs = model(images)
                    if outputs.size(2) != labels.size(1) or outputs.size(3) != labels.size(2):
                        labels = F.interpolate(labels.unsqueeze(1).float(), size=outputs.size()[2:],
                                               mode='nearest').squeeze(1)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                self.train_progress.append(avg_loss)
                self.epoch_numbers.append(epoch + 1)
                self.update_training_curve()
                self.log_message(f"Epoch {epoch + 1}/{epochs} completed. Loss: {avg_loss:.4f}")

            # Save trained model
            model_save_path = os.path.join(save_address, f"{self.selected_model}_model.pth")
            torch.save(model.state_dict(), model_save_path)
            self.log_message(f"Model saved to {model_save_path}")

            # Testing
            self.run_testing_process(test_loader, save_address)

        except Exception as e:
            self.log_message(f"An error occurred during training: {e}")

    def update_training_curve(self):
        """
        Update the training curve displayed on the GUI.
        """
        if not self.epoch_numbers or not self.train_progress:
            return

        self.ax.clear()
        self.ax.plot(self.epoch_numbers, self.train_progress, marker='o', linestyle='-', color='b')
        self.ax.set_title("Training Curve")
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True)

        self.canvas.draw()

    def run_testing_process(self, test_loader, save_address):
        try:
            model = self.models[self.selected_model]
            device = torch.device("cpu")  # Explicitly set the device to CPU
            model = model.to(device)  # Move the model to the CPU
            model.eval()

            save_results_path = os.path.join(save_address, "test_results")
            os.makedirs(save_results_path, exist_ok=True)

            self.log_message("Testing started...")
            results = []

            with torch.no_grad():
                for i, (images, labels) in enumerate(test_loader):
                    images = images.to(device)
                    labels = labels.squeeze(1)  # Ensure labels are in [B, H, W] format
                    outputs = model(images)
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()

                    for idx, pred in enumerate(predictions):
                        result_image = (pred * 255).astype('uint8')
                        result_path = os.path.join(save_results_path, f"result_{i * len(predictions) + idx + 1}.png")
                        cv2.imwrite(result_path, result_image)
                        results.append(result_image)

            self.display_test_results(results[:6])  # Display up to 6 results
            self.log_message(f"Test results saved to {save_results_path}")

        except Exception as e:
            self.log_message(f"An error occurred during testing: {e}")

    def display_test_results(self, results):
        """
        Display test results in the result boxes on the GUI.
        """
        for box in self.result_boxes:
            for widget in box.winfo_children():
                widget.destroy()  # Clear previous content

        for box, result in zip(self.result_boxes, results):
            if result is not None:
                image = Image.fromarray(result)
                image = image.resize((320, 220))
                img_tk = ImageTk.PhotoImage(image)
                label = tk.Label(box, image=img_tk)
                label.image = img_tk
                label.pack()

    def log_message(self, message):
        """
        Append a log message to the log area.
        """
        self.log_area.configure(state="normal")
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.configure(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalImageSegmentationApp(root)
    root.mainloop()
