import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import ssl
from pathlib import Path
import time
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

    def create_model(self):
        try:
            if self.model_name == "resnet18":
                model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
                num_features = 512
            else:  # resnet50
                model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
                num_features = 2048

            # Modify final layer for binary classification
            model.fc = nn.Sequential(nn.Linear(num_features, 1), nn.Sigmoid())
            return model

        except Exception as e:
            print(f"Error loading model: {e}")
            raise


def calculate_class_weights(labels, device):
    n_samples = len(labels)
    n_classes = 2  # binary classification

    # Count samples in each class
    count_no = labels.count(0)
    count_yes = labels.count(1)

    # Calculate weights: higher weight for minority class
    weight_yes = n_samples / (n_classes * count_yes)
    weight_no = n_samples / (n_classes * count_no)

    print(f"Class weights - No: {weight_no:.2f}, Yes: {weight_yes:.2f}")
    return torch.tensor([weight_no, weight_yes], device=device)


def load_dataset(base_path):
    """
    Load images from a directory structure:
    base_path/
        ├── train/
        │   ├── yes/
        │   └── no/
        └── val/
            ├── yes/
            └── no/
    """
    base_path = Path(base_path)
    datasets = {}

    for split in ["train", "val"]:
        image_paths = []
        labels = []

        # Load 'yes' images
        yes_path = base_path / split / "yes"
        if yes_path.exists():
            for img_path in yes_path.glob("*.[jJ][pP][gG]"):
                image_paths.append(str(img_path))
                labels.append(1)
            for img_path in yes_path.glob("*.[pP][nN][gG]"):
                image_paths.append(str(img_path))
                labels.append(1)

        # Load 'no' images
        no_path = base_path / split / "no"
        if no_path.exists():
            for img_path in no_path.glob("*.[jJ][pP][gG]"):
                image_paths.append(str(img_path))
                labels.append(0)
            for img_path in no_path.glob("*.[pP][nN][gG]"):
                image_paths.append(str(img_path))
                labels.append(0)

        datasets[split] = (image_paths, labels)

        print(f"{split} set:")
        print(f"  Total images: {len(image_paths)}")
        print(f"  Yes images: {labels.count(1)}")
        print(f"  No images: {labels.count(0)}\n")

    return datasets


def train_model(base_path, model_name="resnet18", num_epochs=10, batch_size=32):
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
            transforms.Resize(224, antialias=True),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    # Load datasets
    datasets = load_dataset(base_path)
    if not datasets["train"][0] or not datasets["val"][0]:
        raise ValueError("No images found in training or validation directories")

    # Calculate class weights
    class_weights = calculate_class_weights(datasets["train"][1], device)

    # Create data loaders
    train_dataset = ImageDataset(datasets["train"][0], datasets["train"][1], transform)
    val_dataset = ImageDataset(datasets["val"][0], datasets["val"][1], transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and training
    trainer = ModelTrainer(model_name, device)
    criterion = nn.BCELoss(
        reduction="none"
    )  # Changed to 'none' to apply weights per sample
    optimizer = optim.Adam(trainer.model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

    best_val_acc = 0
    metrics = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": [],
    }

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training phase
        trainer.model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = trainer.model(inputs).squeeze()

            # Apply class weights to loss
            loss = criterion(outputs, labels)
            weighted_loss = loss * class_weights[labels.long()]
            loss = weighted_loss.mean()

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation phase
        trainer.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = trainer.model(inputs).squeeze()

                # Apply class weights to validation loss too
                loss = criterion(outputs, labels)
                weighted_loss = loss * class_weights[labels.long()]
                loss = weighted_loss.mean()

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        epoch_time = time.time() - epoch_start

        # Store metrics
        metrics["train_loss"].append(epoch_train_loss)
        metrics["train_acc"].append(epoch_train_acc)
        metrics["val_loss"].append(epoch_val_loss)
        metrics["val_acc"].append(epoch_val_acc)
        metrics["epoch_times"].append(epoch_time)

        # Update learning rate
        scheduler.step(epoch_val_loss)

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(trainer.model.state_dict(), f"best_{model_name}_model.pth")

        # Print progress
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        print(f"Epoch time: {epoch_time:.1f}s")

    return metrics


if __name__ == "__main__":
    # Example usage
    metrics = train_model(
        base_path="data",  # Your base directory containing train and val folders
        model_name="resnet18",
        num_epochs=10,
        batch_size=64,
    )
