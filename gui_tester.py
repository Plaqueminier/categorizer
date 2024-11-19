import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import os
from pathlib import Path


class ImageClassifier:
    def __init__(self, model_path, model_type="resnet18"):
        # Set device
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Create model
        if model_type.lower() == "resnet18":
            self.model = models.resnet18(
                weights=None
            )  # Fixed: removed pretrained parameter
            self.model.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        else:  # resnet50
            self.model = models.resnet50(
                weights=None
            )  # Fixed: removed pretrained parameter
            self.model.fc = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())

        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Define image transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, image_path):
        """Predict a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                output = self.model(input_tensor)

            # Get prediction
            probability = output.item()
            prediction = "YES" if probability > 0.5 else "NO"
            confidence = probability if probability > 0.5 else (1 - probability)

            return {
                "prediction": prediction,
                "confidence": confidence * 100,  # Convert to percentage
                "raw_probability": probability,
            }

        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")


class ImageClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")

        # Set minimum window size
        self.root.minsize(500, 400)

        # Initialize model as None
        self.classifier = None
        self.current_image = None

        # Create GUI elements
        self.create_widgets()

        # Configure grid weights
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(2, weight=1)

    def create_widgets(self):
        # Model loading frame
        model_frame = ttk.LabelFrame(self.root, text="Model Selection", padding="5")
        model_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        ttk.Button(model_frame, text="Load Model", command=self.load_model).pack(
            side=tk.LEFT, padx=5
        )
        self.model_label = ttk.Label(model_frame, text="No model loaded")
        self.model_label.pack(side=tk.LEFT, padx=5)

        # Model type selection
        self.model_type = tk.StringVar(value="resnet18")
        ttk.Radiobutton(
            model_frame, text="ResNet18", variable=self.model_type, value="resnet18"
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            model_frame, text="ResNet50", variable=self.model_type, value="resnet50"
        ).pack(side=tk.LEFT, padx=5)

        # Image selection frame
        image_frame = ttk.LabelFrame(self.root, text="Image Selection", padding="5")
        image_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        ttk.Button(image_frame, text="Select Image", command=self.load_image).pack(
            side=tk.LEFT, padx=5
        )
        self.image_label = ttk.Label(image_frame, text="No image selected")
        self.image_label.pack(side=tk.LEFT, padx=5)

        # Results frame
        results_frame = ttk.LabelFrame(self.root, text="Results", padding="10")
        results_frame.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

        # Image display
        self.image_display = ttk.Label(results_frame)
        self.image_display.pack(pady=10)

        # Prediction display
        self.prediction_label = ttk.Label(
            results_frame, text="", font=("Arial", 12, "bold")
        )
        self.prediction_label.pack(pady=5)

        self.confidence_label = ttk.Label(results_frame, text="")
        self.confidence_label.pack(pady=5)

    def load_model(self):
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")],
        )

        if model_path:
            try:
                self.classifier = ImageClassifier(model_path, self.model_type.get())
                self.model_label.config(text=os.path.basename(model_path))
                messagebox.showinfo(
                    "Success",
                    f"Model loaded successfully\nUsing device: {self.classifier.device}",
                )

                if self.current_image:
                    self.classify_image()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def load_image(self):
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("PNG Files", "*.png"),
                ("All Files", "*.*"),
            ],
        )

        if image_path:
            try:
                # Store image path
                self.current_image = image_path
                self.image_label.config(text=os.path.basename(image_path))

                # Display image
                image = Image.open(image_path).convert("RGB")
                # Calculate scaling to maintain aspect ratio
                display_size = (300, 300)
                image.thumbnail(display_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.image_display.config(image=photo)
                self.image_display.image = photo  # Keep a reference

                # Classify if model is loaded
                if self.classifier:
                    self.classify_image()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def classify_image(self):
        if not self.classifier or not self.current_image:
            return

        try:
            result = self.classifier.predict(self.current_image)

            self.prediction_label.config(
                text=f"Prediction: {result['prediction']}",
                foreground="green" if result["prediction"] == "YES" else "red",
            )
            self.confidence_label.config(
                text=f"Confidence: {result['confidence']:.1f}%\n"
                f"Raw Probability: {result['raw_probability']:.4f}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to classify image: {str(e)}")


def main():
    root = tk.Tk()
    app = ImageClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
