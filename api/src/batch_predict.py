import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import sys
import os


class TransferModel(nn.Module):
    def __init__(self, pretrained=False):
        super(TransferModel, self).__init__()

        # Use ResNet18 with same architecture as training
        self.resnet = models.resnet18(pretrained=pretrained)

        # Add batch normalization and dropout layers
        self.resnet.layer1 = nn.Sequential(
            self.resnet.layer1, nn.BatchNorm2d(64), nn.Dropout(0.1)
        )
        self.resnet.layer2 = nn.Sequential(
            self.resnet.layer2, nn.BatchNorm2d(128), nn.Dropout(0.2)
        )
        self.resnet.layer3 = nn.Sequential(
            self.resnet.layer3, nn.BatchNorm2d(256), nn.Dropout(0.2)
        )

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.resnet(x)


class ImageProcessor:
    def __init__(self, model_path: str):
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = TransferModel()

        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different saving formats
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # Define transforms - same as training
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Restored to original size
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def predict(self, image_path: str):
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            output = self.model(image_tensor)
            probability = output.squeeze().item()

            return probability
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def process_directory(self, directory: str):
        image_dir = Path(directory)
        if not image_dir.exists():
            print(f"Directory {directory} not found!")
            return

        image_files = [
            f
            for f in image_dir.glob("*")
            if f.is_file()
            and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            and not f.name.startswith(".")
        ]

        if not image_files:
            print("No images found in directory!")
            return

        print(f"Processing {len(image_files)} images...")

        for img_path in image_files:
            probability = self.predict(str(img_path))
            if probability is None:
                continue

            # Convert probability to percentage string
            if probability > 0.5:
                percentage = min(99, int(probability * 100))
            else:
                percentage = max(0, int(probability * 100))

            new_name = f"{percentage:02d}_{img_path.name}"
            new_path = img_path.parent / new_name

            try:
                os.rename(img_path, new_path)
                print(
                    f"Renamed {img_path.name} -> {new_name} (confidence: {probability:.2%})"
                )
            except Exception as e:
                print(f"Error renaming {img_path.name}: {str(e)}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_predict.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]

    if not Path(model_path).exists():
        print(f"Model file {model_path} not found!")
        sys.exit(1)

    try:
        processor = ImageProcessor(model_path)
        processor.process_directory("images")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
