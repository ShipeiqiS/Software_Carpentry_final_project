import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

        # Layout setup
        self.setup_layout(window_width, window_height)

    def setup_layout(self, window_width, window_height):
        # Title
        title_label = tk.Label(self.root, text="Medical Image Segmentation Expert", font=("Arial", 18, "bold"))
        title_label.pack(pady=10)

        # Batch size, Total epochs, and Learning rate
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        tk.Label(control_frame, text="Batch Size:", font=("Arial", 12)).grid(row=0, column=0, padx=5)
        tk.Button(control_frame, text="-", command=self.decrease_batch_size, font=("Arial", 12)).grid(row=0, column=1, padx=5)
        self.batch_size_label = tk.Label(control_frame, text="5", width=5, relief=tk.SUNKEN, font=("Arial", 12))
        self.batch_size_label.grid(row=0, column=2, padx=5)
        tk.Button(control_frame, text="+", command=self.increase_batch_size, font=("Arial", 12)).grid(row=0, column=3, padx=5)

        tk.Label(control_frame, text="Total Epochs:", font=("Arial", 12)).grid(row=0, column=4, padx=5)
        tk.Button(control_frame, text="-", command=self.decrease_epochs, font=("Arial", 12)).grid(row=0, column=5, padx=5)
        self.epochs_label = tk.Label(control_frame, text="100", width=5, relief=tk.SUNKEN, font=("Arial", 12))
        self.epochs_label.grid(row=0, column=6, padx=5)
        tk.Button(control_frame, text="+", command=self.increase_epochs, font=("Arial", 12)).grid(row=0, column=7, padx=5)

        tk.Label(control_frame, text="Learning Rate:", font=("Arial", 12)).grid(row=0, column=8, padx=5)
        tk.Button(control_frame, text="-", command=self.decrease_learning_rate, font=("Arial", 12)).grid(row=0, column=9, padx=5)
        self.learning_rate_label = tk.Label(control_frame, text="0.02", width=5, relief=tk.SUNKEN, font=("Arial", 12))
        self.learning_rate_label.grid(row=0, column=10, padx=5)
        tk.Button(control_frame, text="+", command=self.increase_learning_rate, font=("Arial", 12)).grid(row=0, column=11, padx=5)

        # Run button
        run_button = tk.Button(self.root, text="Run", font=("Arial", 14, "bold"), bg="green", fg="white", command=self.run_segmentation)
        run_button.pack(pady=10)

        # Model selection
        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=10)
        tk.Label(model_frame, text="Select Model:", font=("Arial", 12)).grid(row=0, column=0, padx=5)
        for i in range(4):
            tk.Button(model_frame, text=f"Model {i + 1}", font=("Arial", 12), command=lambda idx=i: self.select_model(idx)).grid(row=0, column=i + 1, padx=5)

        # Dataset information
        dataset_frame = tk.Frame(self.root)
        dataset_frame.pack(pady=10)

        tk.Label(dataset_frame, text="Train Dataset:", font=("Arial", 12)).grid(row=0, column=0, padx=5, sticky="e")
        tk.Entry(dataset_frame, width=40).grid(row=0, column=1, padx=5)

        tk.Label(dataset_frame, text="Train Label:", font=("Arial", 12)).grid(row=1, column=0, padx=5, sticky="e")
        tk.Entry(dataset_frame, width=40).grid(row=1, column=1, padx=5)

        tk.Label(dataset_frame, text="Test Dataset:", font=("Arial", 12)).grid(row=2, column=0, padx=5, sticky="e")
        tk.Entry(dataset_frame, width=40).grid(row=2, column=1, padx=5)

        tk.Label(dataset_frame, text="Test Label:", font=("Arial", 12)).grid(row=3, column=0, padx=5, sticky="e")
        tk.Entry(dataset_frame, width=40).grid(row=3, column=1, padx=5)

        # Main layout: Train curve and results
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Epoch train curve display
        train_curve_frame = tk.Frame(main_frame, relief=tk.SUNKEN, bd=2, bg="white")
        train_curve_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        tk.Label(train_curve_frame, text="Epoch Train Curve", font=("Arial", 14, "bold"), bg="white").pack(pady=10)

        figure = plt.Figure(figsize=(6, 5), dpi=100)  # Adjusted dimensions
        ax = figure.add_subplot(111)
        ax.grid(True)
        ax.set_title("Training Curve")
        canvas = FigureCanvasTkAgg(figure, train_curve_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Sample results
        results_frame = tk.Frame(main_frame, relief=tk.SUNKEN, bd=2, bg="white")
        results_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        tk.Label(results_frame, text="Sample Results", font=("Arial", 14, "bold"), fg="blue", bg="white").pack(pady=10)

        result_boxes_frame = tk.Frame(results_frame, bg="white")
        result_boxes_frame.pack(fill=tk.BOTH, expand=True)

        # Adjust to 2x3 layout with larger boxes
        for i in range(2):
            for j in range(3):
                result_box = tk.Frame(result_boxes_frame, width=320, height=220, relief=tk.SUNKEN, bd=2, bg="white")  # Increased dimensions
                result_box.grid(row=i, column=j, padx=20, pady=20)

        # Configure row/column weights
        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=2)
        main_frame.grid_rowconfigure(0, weight=1)


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

    def select_model(self, model_index):
        messagebox.showinfo("Model Selection", f"Model {model_index + 1} selected.")

    def run_segmentation(self):
        messagebox.showinfo("Run", "Segmentation process started.")

    def decrease_learning_rate(self):
        current_rate = float(self.learning_rate_label["text"])
        if current_rate > 0.01:
            self.learning_rate_label["text"] = f"{current_rate - 0.01:.2f}"

    def increase_learning_rate(self):
        current_rate = float(self.learning_rate_label["text"])
        self.learning_rate_label["text"] = f"{current_rate + 0.01:.2f}"

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalImageSegmentationApp(root)
    root.mainloop()
