from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import shutil
from pathlib import Path

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/images", StaticFiles(directory="images"), name="images")


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


class ImageRequest(BaseModel):
    filename: str
    category: str


class Classifier:
    def __init__(self, model_path: str):
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")
        print("Loading transfer learning model...")

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

        # Transform - same as training
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Restored to original size
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Warm up
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            self.model(dummy_input)

    @torch.no_grad()
    def predict(self, image_path: str):
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            output = self.model(image_tensor)
            probability = output.squeeze().item()

            prediction = "yes" if probability > 0.5 else "no"
            confidence = probability if probability > 0.5 else 1 - probability

            return prediction, confidence

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# Initialize classifier with the transfer learning model
classifier = Classifier(
    model_path="./models/best_model.pth"  # Update this path to your model file
)


@app.get("/api/image")
async def get_image():
    try:
        images_dir = Path("images")
        if not images_dir.exists():
            raise HTTPException(status_code=404, detail="No images directory found")

        # Get all image files and sort them alphabetically
        image_files = sorted(
            [
                f
                for f in images_dir.glob("*")
                if f.is_file() and not f.name.startswith(".")
            ],
            key=lambda x: x.name,
        )

        if not image_files:
            raise HTTPException(status_code=404, detail="No images available")

        first_image = image_files[0]
        prediction, confidence = classifier.predict(str(first_image))

        return {
            "filename": first_image.name,
            "url": f"/images/{first_image.name}",
            "prediction": prediction,
            "confidence": confidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/categorize")
async def categorize_image(request: ImageRequest):
    try:
        if request.category not in ["yes", "no"]:
            raise HTTPException(status_code=400, detail="Invalid category")

        source_path = Path("images") / request.filename
        if not source_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        target_dir = Path(request.category)
        target_dir.mkdir(exist_ok=True)
        target_path = target_dir / request.filename

        shutil.move(str(source_path), str(target_path))
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/image/count")
async def get_image_count():
    try:

        def count_images(directory: str) -> int:
            path = Path(directory)
            if not path.exists():
                return 0
            return len(
                [
                    f
                    for f in path.glob("*")
                    if f.is_file() and not f.name.startswith(".")
                ]
            )

        return {
            "remaining": count_images("images"),
            "yes": count_images("yes"),
            "no": count_images("no"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3009)
