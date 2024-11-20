import torch
import torchvision.models as models
from torch import nn
import onnx


class BinaryResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the pretrained ResNet18 model
        self.resnet = models.resnet18(pretrained=False)

        # Modify the final layer for binary classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Linear(num_features, 1), nn.Sigmoid())

    def forward(self, x):
        return self.resnet(x)


# Load your trained model
model = BinaryResNet18()
state_dict = torch.load("resnet18_binary_classifier.pth", map_location=torch.device("cpu"))
# Add 'resnet.' prefix to all keys
new_state_dict = {'resnet.' + k: v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

# Create dummy input with the correct shape
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model
torch.onnx.export(
    model,  # model being run
    dummy_input,  # model input (or a tuple for multiple inputs)
    "resnet18.onnx",  # where to save the model
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=12,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={
        "input": {0: "batch_size"},  # variable length axes
        "output": {0: "batch_size"},
    },
)

print("Model converted to ONNX format successfully!")

onnx_model = onnx.load("resnet18.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")
