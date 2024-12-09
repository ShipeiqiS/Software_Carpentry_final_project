import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import time
import cv2
import random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import threading


class MedicalImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Image Segmentation Expert")

        # Set the window size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        self.root.geometry(f"{window_width}x{window_height}")

        # Current selected model
        self.selected_model = None
        self.train_progress = []
        self.epoch_numbers = []

        # Layout setup
        self.setup_layout(window_width, window_height)

    def setup_layout(self, window_width, window_height):
        # Title
        title_label = tk.Label(self.root, text="Medical Image Segmentation Expert", font=("Arial", 18, "bold"))
        title_label.pack(pady=10)

        # Control Panel
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        # Add batch, epoch, and learning rate controls
        self.add_control_buttons(control_frame)

        # Save Address input
        self.add_save_address(control_frame)

        # Model selection buttons
        self.add_model_selection()

        # Dataset information inputs
        self.add_dataset_inputs()

        # Log area
        self.add_log_area()

        # Training curve and result display
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
        current_size = int(self.batch_size_label["text"])
        if current_size > 1:
            self.batch_size_label["text"] = str(current_size - 1)

    def increase_batch_size(self):
        current_size = int(self.batch_size_label["text"])
        self.batch_size_label["text"] = str(current_size + 1)

    def decrease_epochs(self):
        current_epochs = int(self.epochs_label["text"])
        if current_epochs > 1:
            self.epochs_label["text"] = str(current_epochs - 1)

    def increase_epochs(self):
        current_epochs = int(self.epochs_label["text"])
        self.epochs_label["text"] = str(current_epochs + 1)

    def decrease_learning_rate(self):
        current_rate = float(self.learning_rate_label["text"])
        if current_rate > 0.01:
            self.learning_rate_label["text"] = f"{current_rate - 0.01:.2f}"

    def increase_learning_rate(self):
        current_rate = float(self.learning_rate_label["text"])
        self.learning_rate_label["text"] = f"{current_rate + 0.01:.2f}"

    def add_save_address(self, frame):
        tk.Label(frame, text="Save Address:", font=("Arial", 12)).grid(row=1, column=0, padx=5)
        self.save_address_entry = tk.Entry(frame, width=50)
        self.save_address_entry.grid(row=1, column=1, columnspan=10, padx=5, pady=10)

        run_button = tk.Button(self.root, text="Run", font=("Arial", 14, "bold"), bg="green", fg="white",
                               command=self.start_training)
        run_button.pack(pady=10)

    def add_model_selection(self):
        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=10)
        tk.Label(model_frame, text="Select Model:", font=("Arial", 12)).grid(row=0, column=0, padx=5)

        models = ["U-Net", "SwinUnet", "Deeplabv3Plus", "FCN-ResNet"]
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
                self.selected_model = btn["text"]
            else:
                btn.config(bg="SystemButtonFace", fg="black")

    def add_dataset_inputs(self):
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
        log_frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=2)
        log_frame.pack(fill=tk.BOTH, expand=False, padx=20, pady=10)
        tk.Label(log_frame, text="Logs", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        self.log_area = scrolledtext.ScrolledText(log_frame, width=100, height=10, state="disabled")
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def log_message(self, message):
        self.log_area.configure(state="normal")
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.configure(state="disabled")

    def add_result_area(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Epoch train curve display
        train_curve_frame = tk.Frame(main_frame, relief=tk.SUNKEN, bd=2, bg="white")
        train_curve_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        tk.Label(train_curve_frame, text="Epoch Train Curve", font=("Arial", 14, "bold"), bg="white").pack(pady=10)

        figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax = figure.add_subplot(111)
        self.ax.grid(True)
        self.ax.set_title("Training Curve")
        self.canvas = FigureCanvasTkAgg(figure, train_curve_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Sample results
        results_frame = tk.Frame(main_frame, relief=tk.SUNKEN, bd=2, bg="white")
        results_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        tk.Label(results_frame, text="Sample Results", font=("Arial", 14, "bold"), fg="blue", bg="white").pack(pady=10)

        self.result_boxes_frame = tk.Frame(results_frame, bg="white")
        self.result_boxes_frame.pack(fill=tk.BOTH, expand=True)
        self.result_boxes = []
        for i in range(2):
            for j in range(3):
                result_box = tk.Frame(self.result_boxes_frame, width=320, height=220, relief=tk.SUNKEN, bd=2, bg="white")
                result_box.grid(row=i, column=j, padx=20, pady=20)
                self.result_boxes.append(result_box)

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
            elif not os.listdir(path):
                messagebox.showerror("Error", f"{label} path is empty: {path}")
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

        threading.Thread(target=self.run_training_process, args=(batch_size, epochs, learning_rate)).start()

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
        self.ax.set_ylabel("Accuracy")
        self.ax.grid(True)

        self.canvas.draw()

    def run_training_process(self, batch_size, epochs, learning_rate):
        try:
            for epoch in range(epochs):
                time.sleep(0.5)  # Simulate training time
                self.train_progress.append(random.uniform(0.7, 0.95))
                self.epoch_numbers.append(epoch + 1)
                self.update_training_curve()
                self.log_message(f"Epoch {epoch + 1}/{epochs} completed.")
        except Exception as e:
            self.log_message(f"Error occurred: {e}")

    def run_testing_process(self, test_data, save_address):
        """Simulate testing process and generate test results."""
        results = []
        test_images = os.listdir(test_data)
        for i in range(min(6, len(test_images))):  # Get up to 6 results
            test_image_path = os.path.join(test_data, random.choice(test_images))
            image = cv2.imread(test_image_path)
            if image is not None:
                results.append(image)
        return results

    def display_test_results(self, results):
        """Display test results in the result boxes."""
        for box in self.result_boxes:
            for widget in box.winfo_children():
                widget.destroy()  # Clear previous content

        for box, result in zip(self.result_boxes, results):
            if result is not None:
                image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                image = image.resize((320, 220))
                img_tk = ImageTk.PhotoImage(image)
                label = tk.Label(box, image=img_tk)
                label.image = img_tk
                label.pack()

    def save_model_checkpoint(self, save_path, model_name):
        """Save the trained model checkpoint to the specified path."""
        try:
            os.makedirs(save_path, exist_ok=True)
            checkpoint_file = os.path.join(save_path, f"{model_name}_checkpoint.pth")
            with open(checkpoint_file, 'w') as f:
                f.write("Simulated Model Checkpoint")  # Replace with actual save logic
            self.log_message(f"Model checkpoint saved to: {checkpoint_file}")
        except Exception as e:
            self.log_message(f"Error saving model checkpoint: {e}")

    def save_test_results(self, results, save_path):
        """Save test results to the specified path."""
        try:
            os.makedirs(save_path, exist_ok=True)
            for idx, result in enumerate(results):
                result_path = os.path.join(save_path, f"test_result_{idx + 1}.png")
                cv2.imwrite(result_path, result)
                self.log_message(f"Test result saved to: {result_path}")
        except Exception as e:
            self.log_message(f"Error saving test results: {e}")

    def run_complete_workflow(self):
        """Complete workflow from validating input, training, testing, and saving results."""
        try:
            # Validate paths
            train_data = self.train_dataset_entry.get()
            train_label = self.train_label_entry.get()
            test_data = self.test_dataset_entry.get()
            save_address = self.save_address_entry.get()

            if not self.validate_paths():
                return

            if not self.selected_model:
                messagebox.showerror("Error", "Please select a model before running.")
                return

            batch_size = int(self.batch_size_label["text"])
            epochs = int(self.epochs_label["text"])
            learning_rate = float(self.learning_rate_label["text"])

            # Log configuration
            self.log_message(f"Model: {self.selected_model}")
            self.log_message(f"Batch Size: {batch_size}")
            self.log_message(f"Epochs: {epochs}")
            self.log_message(f"Learning Rate: {learning_rate}")

            # Simulate training
            self.log_message("Training started...")
            self.run_training_process(batch_size, epochs, learning_rate)

            # Save trained model
            self.save_model_checkpoint(save_address, self.selected_model)

            # Simulate testing
            self.log_message("Testing started...")
            test_results = self.run_testing_process(test_data, save_address)
            self.log_message("Testing completed.")

            # Save and display test results
            self.save_test_results(test_results, os.path.join(save_address, "test_results"))
            self.display_test_results(test_results)

            self.log_message("Workflow completed successfully.")
        except Exception as e:
            self.log_message(f"An unexpected error occurred: {e}")
            messagebox.showerror("Error", str(e))

    def reset_gui(self):
        """Reset all fields and selections in the GUI."""
        self.batch_size_label.config(text="5")
        self.epochs_label.config(text="100")
        self.learning_rate_label.config(text="0.02")
        self.train_dataset_entry.delete(0, tk.END)
        self.train_label_entry.delete(0, tk.END)
        self.test_dataset_entry.delete(0, tk.END)
        self.test_label_entry.delete(0, tk.END)
        self.save_address_entry.delete(0, tk.END)
        self.log_area.configure(state="normal")
        self.log_area.delete("1.0", tk.END)
        self.log_area.configure(state="disabled")
        self.selected_model = None
        for btn in self.model_buttons:
            btn.config(bg="SystemButtonFace", fg="black")
        for box in self.result_boxes:
            for widget in box.winfo_children():
                widget.destroy()
        self.log_message("GUI reset to default values.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalImageSegmentationApp(root)
    root.mainloop()
