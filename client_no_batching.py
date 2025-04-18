import numpy as np
import tritonclient.http as httpclient
import time

MODEL_NAME = "resnet18_onnx"
TRITON_URL = "localhost:8000"

def create_input_data(batch_size):
    # Create randow data simulating batched images (Batch, Channel, Height, Width)
    # Important: Shape must match the DIMS expected by the config.pbtxt, INCLUDING batch
    input_np = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
    input_tensor = httpclient.InferInput("input", input_np.shape, "FP32")
    input_tensor.set_data_from_numpy(input_np, binary_data=True)
    return input_tensor


# Connect to triton and create a client
try:
    triton_client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
except Exception as e:
    print(f"Could not creat Triton client: {e}")
    exit(1)


if not triton_client.is_server_live():
    print(f"Triton server at {TRITON_URL} is not live.")
    exit(1)

if not triton_client.is_model_ready(MODEL_NAME):
    print(f"Model {MODEL_NAME} is not ready on Triton Server")
    exit(1)


# Test 1:  Sending a single request
print("\n--- Sending single request (Batch Size: 1) ---")
start_time = time.time()
single_input = create_input_data(batch_size=1)
try:
    results = triton_client.infer(
            model_name=MODEL_NAME,
            inputs=[single_input],
            outputs=[httpclient.InferRequestedOutput("output", binary_data=True)]
            )
    output_data = results.as_numpy("output")
    print(f"Received output shape: {output_data.shape}")
except Exception as e:
    print(f"Inference failed: {e}")

end_time = time.time()
print(f"Single request latency: {end_time - start_time:.4f} seconds")


# Test 2: Send an EXPLICIT BATCH request
print("\n--- Sending explicit batch request (Batch Size: 4) ---")
BATCH_SIZE_CLIENT=4
start_time = time.time()
batch_input = create_input_data(batch_size=BATCH_SIZE_CLIENT)
try:
    results = triton_client.infer(
            model_name=MODEL_NAME,
            inputs=[batch_input],
            outputs=[httpclient.InferRequestedOutput("output", binary_data=True)]
            )
    output_data = results.as_numpy("output")
    print(f"Received output shape: {output_data.shape}")
except Exception as e:
    print(f"Inference failed: {e}")

end_time = time.time()
print(f"Explicit batch ({BATCH_SIZE_CLIENT}) latency: {end_time - start_time:.4f} seconds")
print(f"Latency per item in batch: {(end_time - start_time) / BATCH_SIZE_CLIENT:.4f} seconds")

print("\n Finished tests for max_batch_size: 0")
