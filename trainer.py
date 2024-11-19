import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import ssl
import certifi
from sklearn.model_selection import train_test_split
import time
import numpy as np
from torchvision.models import ResNet18_Weights, ResNet50_Weights

# SSL Certificate fix for macOS
ssl._create_default_https_context = ssl._create_unverified_context


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


class ModelTrainer:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.model = self.create_model()
        self.model.to(device)

        # Calculate model size
        self.model_size = (
            sum(p.numel() for p in self.model.parameters()) / 1e6
        )  # Size in millions

    def create_model(self):
        try:
            if self.model_name == "resnet18":
                # Try loading with SSL verification first
                model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            else:  # resnet50
                model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        except Exception as e:
            print(f"Warning: Error loading pretrained weights: {e}")
            print("Attempting to load model without pretrained weights...")
            if self.model_name == "resnet18":
                model = models.resnet18(pretrained=False)
            else:  # resnet50
                model = models.resnet50(pretrained=False)

        # Modify final layer for binary classification
        if self.model_name == "resnet18":
            model.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        else:  # resnet50 has a different number of features
            model.fc = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())
        return model

    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_times = []

        for inputs, labels in train_loader:
            start_time = time.time()

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_time = time.time() - start_time
            batch_times.append(batch_time)

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return (
            running_loss / len(train_loader),
            100 * correct / total,
            np.mean(batch_times),
        )

    def validate(self, val_loader, criterion):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_times = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                start_time = time.time()

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs).squeeze()
                loss = criterion(outputs, labels)

                batch_time = time.time() - start_time
                batch_times.append(batch_time)

                running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return (
            running_loss / len(val_loader),
            100 * correct / total,
            np.mean(batch_times),
        )


def compare_models(yes_folder, no_folder, num_epochs=10):
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load and split data
    image_paths = [
        (os.path.join(yes_folder, f), 1)
        for f in os.listdir(yes_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ] + [
        (os.path.join(no_folder, f), 0)
        for f in os.listdir(no_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_paths:
        raise ValueError(f"No images found in {yes_folder} or {no_folder}")

    print(f"Total images found: {len(image_paths)}")
    print(f"Yes images: {sum(1 for _, label in image_paths if label == 1)}")
    print(f"No images: {sum(1 for _, label in image_paths if label == 0)}")

    paths, labels = zip(*image_paths)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = ImageDataset(train_paths, train_labels, transform)
    val_dataset = ImageDataset(val_paths, val_labels, transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    results = {}
    for model_name in ["resnet18", "resnet50"]:
        print(f"\nTraining {model_name.upper()}...")

        trainer = ModelTrainer(model_name, device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(trainer.model.parameters(), lr=0.001)

        model_metrics = {
            "train_losses": [],
            "train_accs": [],
            "val_losses": [],
            "val_accs": [],
            "train_times": [],
            "val_times": [],
            "model_size": trainer.model_size,
        }

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_loss, train_acc, train_time = trainer.train_epoch(
                train_loader, criterion, optimizer
            )
            model_metrics["train_losses"].append(train_loss)
            model_metrics["train_accs"].append(train_acc)
            model_metrics["train_times"].append(train_time)

            val_loss, val_acc, val_time = trainer.validate(val_loader, criterion)
            model_metrics["val_losses"].append(val_loss)
            model_metrics["val_accs"].append(val_acc)
            model_metrics["val_times"].append(val_time)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(
                f"Batch times - Train: {train_time * 1000:.1f}ms, Val: {val_time * 1000:.1f}ms"
            )

        # Save model
        torch.save(trainer.model.state_dict(), f"{model_name}_binary_classifier.pth")
        results[model_name] = model_metrics

    return results


if __name__ == "__main__":
    results = compare_models("api/yes", "api/no")
