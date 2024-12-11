import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import numpy as np
import ssl
from torchvision.models import ResNet18_Weights

# Add SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["no", "yes"]

        self.image_paths = []
        self.labels = []

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB for ResNet
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class TransferModel(nn.Module):
    def __init__(self, weights=ResNet18_Weights.DEFAULT):
        super(TransferModel, self).__init__()

        # Use ResNet18 with specified weights
        self.resnet = models.resnet18(weights=weights)

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
            nn.Linear(num_ftrs, 512),  # Increased size
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),  # Increased size
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.resnet(x)


def prepare_data(source_yes_dir, source_no_dir, output_dir):
    for split in ["train", "val", "test"]:
        for label in ["yes", "no"]:
            os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)

    yes_images = [os.path.join(source_yes_dir, f) for f in os.listdir(source_yes_dir)]
    train_yes, temp_yes = train_test_split(yes_images, train_size=0.7, random_state=42)
    val_yes, test_yes = train_test_split(temp_yes, train_size=0.5, random_state=42)

    no_images = [os.path.join(source_no_dir, f) for f in os.listdir(source_no_dir)]
    train_no, temp_no = train_test_split(no_images, train_size=0.7, random_state=42)
    val_no, test_no = train_test_split(temp_no, train_size=0.5, random_state=42)

    for images, target_dir in [
        (train_yes, os.path.join(output_dir, "train", "yes")),
        (train_no, os.path.join(output_dir, "train", "no")),
        (val_yes, os.path.join(output_dir, "val", "yes")),
        (val_no, os.path.join(output_dir, "val", "no")),
        (test_yes, os.path.join(output_dir, "test", "yes")),
        (test_no, os.path.join(output_dir, "test", "no")),
    ]:
        for img in images:
            shutil.copy2(img, target_dir)


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs = inputs.to(device)
        labels = labels.float().to(device)

        # Apply mixup
        inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()

        # Modified loss calculation for mixup
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(
            outputs, labels_b
        )
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(val_loader), 100.0 * correct / total


def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100.0 * correct / total
    return running_loss / len(test_loader), accuracy, all_predictions, all_labels


def main():
    BATCH_SIZE = 16
    NUM_EPOCHS = 100

    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print(f"Using device: {DEVICE}")

    # Simplified data augmentation for faster processing
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Restored to original size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = ImageDataset(root_dir="data/train", transform=train_transform)
    val_dataset = ImageDataset(root_dir="data/val", transform=val_transform)
    test_dataset = ImageDataset(root_dir="data/test", transform=val_transform)

    # Analyze class distribution
    total_samples = len(train_dataset)
    pos_samples = sum(1 for _, label in train_dataset if label == 1)
    neg_samples = total_samples - pos_samples
    pos_ratio = (pos_samples / total_samples) * 100
    neg_ratio = (neg_samples / total_samples) * 100

    print("\nTraining Set Distribution:")
    print(f"Total samples: {total_samples}")
    print(f"Positive samples (yes): {pos_samples} ({pos_ratio:.2f}%)")
    print(f"Negative samples (no): {neg_samples} ({neg_ratio:.2f}%)")
    print(f"Positive/Negative ratio: 1:{neg_samples / pos_samples:.2f}")

    # Calculate and display class weights
    pos_weight = torch.tensor([neg_samples / pos_samples]).to(DEVICE)
    print(f"\nApplying weight to positive class: {pos_weight.item():.2f}")

    print(f"\nValidation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Reduced for CPU
        pin_memory=False,  # Disabled for CPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Reduced for CPU
        pin_memory=False,  # Disabled for CPU
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Reduced for CPU
        pin_memory=False,  # Disabled for CPU
    )

    # Initialize model and training components
    model = TransferModel(weights=ResNet18_Weights.DEFAULT).to(DEVICE)

    criterion = nn.BCELoss(weight=pos_weight)

    # Optimizer with a higher initial learning rate
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0003,  # Adjusted initial learning rate
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    # Scheduler without verbose
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.3,  # Less aggressive reduction factor
        patience=5,  # Reduced patience to adapt quicker
        min_lr=1e-5,  # Higher minimum learning rate
    )

    best_val_acc = 0
    best_model_path = "best_model.pth"

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Learning rates: {scheduler.get_last_lr()}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        # Step the scheduler with validation accuracy
        scheduler.step(val_acc)

        print(f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                best_model_path,
            )
            print(f"Saved new best model with validation accuracy: {val_acc:.2f}%")

    # Load best model for testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Perform final test
    test_loss, test_acc, predictions, labels = test(
        model, test_loader, criterion, DEVICE
    )
    print("\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    SOURCE_YES_DIR = "yes"
    SOURCE_NO_DIR = "no"
    OUTPUT_DIR = "data"

    prepare_data(SOURCE_YES_DIR, SOURCE_NO_DIR, OUTPUT_DIR)
    main()
