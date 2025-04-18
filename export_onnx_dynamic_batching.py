import torch
import torchvision.models as models
import torch.onnx
import os

# --- Load your pre-trained PyTorch model ---
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# --- Define dummy input and paths ---
batch_size = 1
input_channels = 3
height = 224
width = 224
dummy_input = torch.randn(batch_size, input_channels, height, width, requires_grad=False)

onnx_model_dir = "model_repo/resnet18_onnx/2/"
onnx_model_path = os.path.join(onnx_model_dir, "model.onnx")
os.makedirs(onnx_model_dir, exist_ok=True)

# --- Define input/output names ---
input_name = "input"
output_name = "output"

print(f"Exporting ONNX model to: {onnx_model_path}")
print(f"Input name: {input_name}, Output name: {output_name}")


torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=[input_name],
    output_names=[output_name],
    dynamic_axes={
        input_name: {0: 'batch_size'},
        output_name: {0: 'batch_size'}
    }
)

print("ONNX model exported successfully with dynamic axes.")

# --- Verify the export (requires onnxruntime)
import onnxruntime as ort
import numpy as np

sess_options = ort.SessionOptions()
ort_session = ort.InferenceSession(onnx_model_path, sess_options)
input_feed = {input_name: np.random.randn(5, 3, 224, 224).astype(np.float32)} # Test with batch > 1
outputs = ort_session.run([output_name], input_feed)
print("ONNX Runtime check with dynamic batch successful. Output shape:", outputs[0].shape) # Should be (5, 1000)