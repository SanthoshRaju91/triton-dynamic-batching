import torch
import torchvision.models as models
import numpy as np
import os

MODEL_NAME = "resnet18_onnx"
EXPORT_DIR = "model_repo"
MODEL_VERSION = 1
ONNX_FILE_PATH = os.path.join(EXPORT_DIR, MODEL_NAME, str(MODEL_VERSION), "model.onnx")
NUM_CLASSES  = 1000 # Standard for ImageNet trained ResNet18
INPUT_SHAPE_NCHW = (1, 3, 224, 224) # Batch=1, Channel=3, Height=224, Width=224

# Ensure export directory exists
os.makedirs(os.path.dirname(ONNX_FILE_PATH), exist_ok=True)

# Load pre-trained ResNet18 model
print("Loading pre-trained ResNet18 model...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()
print("Model loaded successfully")

# Create dummy input
# We need a dummy input with the correct shape to trace the model during export.
# The batch size here (1) doesn't strictly limit the ONNX model if dynamic_axes is used.
dummy_input = torch.randn(INPUT_SHAPE_NCHW, requires_grad=False)
print(f"Created dummy input with shape: {dummy_input.shape}")


# Export to ONNX
print(f"Exporting model to ONNX at: {ONNX_FILE_PATH}")

try:
    torch.onnx.export(
            model, # the model to export
            dummy_input, # sample input tensor
            ONNX_FILE_PATH, # where to save the ONNX file
            export_params=True, # store the trained parameter weights within the model file
            opset_version=11, # use stable ONNX opset version (11 or higher is good)
            do_constant_folding=True, # Execute constant folding for optimization
            input_names=['input'], # Define a name for the model's input
            output_names=['output'], # Define a name for the model's output
            dynamic_axes={
                "input": {0: "batch_size"}, # Allows the batch dimension (axis 0) of input to vary
                "output": {0: "batch_size"}, # Allows the batch dimension (axis 0) of output to vary
                }
            )
    print("ONNX export successfull")


    # Verify onnx model
    import onnx
    onnx_model = onnx.load(ONNX_FILE_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification successful!")
except Exception as e:
    print(f"Error during ONNX export: {e}")

