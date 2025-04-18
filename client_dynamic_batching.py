import numpy as np
import tritonclient.http as httpclient
import time
import threading

MODEL_NAME = "resnet18_onnx"
TRITON_URL = "localhost:8000"
NUM_CONCURRENT_REQUESTS = 10
# Use the corrected input/output names if you found they were different
INPUT_NAME = "input"
OUTPUT_NAME = "output"


def create_single_input_data():
    input_np = np.random.rand(3, 224, 224).astype(np.float32)
    input_tensor = httpclient.InferInput(INPUT_NAME, input_np.shape, "FP32")
    input_tensor.set_data_from_numpy(input_np, binary_data=True)
    return input_tensor


results_lock = threading.Lock()
results_list = []

def infer_thread_func(thread_id):
    global results_list

    local_triton_client = None

    try:
        local_triton_client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
        single_input = create_single_input_data()
        start_time = time.time()

        result = local_triton_client.infer(
            model_name=MODEL_NAME,
            inputs=[single_input],
            outputs=[httpclient.InferRequestedOutput(OUTPUT_NAME, binary_data=True)]
        )
        output_data = result.as_numpy(OUTPUT_NAME)
        end_time = time.time()

        expected_output_shape = (1000, )

        with results_lock:
            results_list.append({
                "thread_id": thread_id,
                "latency_sec": end_time - start_time,
                "output_shape": output_data.shape
            })
        print(f"Thread {thread_id}: Success, Latency: {end_time - start_time:.4fs}, Output shape: {output_data.shape}")
        if output_data.shape != expected_output_shape:
            print(f"WARNING: Thread {thread_id} - Unexpected output shape. Got {output_data.shape}, Expected {expected_output_shape}")
    except Exception as e:
        print(f"Thread {thread_id}:Inference failed: {e}")
        with results_lock:
            results_list.append({
                "thread_id": thread_id,
                "latency_sec": -1,
                "output_shape": None
            })
    finally:
        if local_triton_client:
            try:
                local_triton_client.close()
            except:
                pass
    
try:
    print("checking server and model status")
    initial_client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)

    if not initial_client.is_server_live():
        print(f"Triton server at {TRITON_URL} is not live. Exiting.")
        exit(1)
    
    if not initial_client.is_model_ready(MODEL_NAME):
        print(f"Model {MODEL_NAME} is not ready on Triton server. Check server logs and config. Exiting.")
        exit(1)

    print(f"Server {TRITON_URL} live, Model {MODEL_NAME} ready. Starting threads...")
    initial_client.close()
except Exception as e:
    print(f"Could not perform initial check on server/model: {e}")
    exit(1)


print(f"\n--- Sending {NUM_CONCURRENT_REQUESTS} concurrent single requests (Dynamic Batching Enabled) ---")
threads = []
total_start_time = time.time()

for i in range(NUM_CONCURRENT_REQUESTS):
    thread = threading.Thread(target=infer_thread_func, args=(i, ))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

total_end_time = time.time()
total_time = total_end_time - total_start_time


print("\n--- Results ---")
successful_requests = [r for r in results_list if r["latency_sec"] >= 0]
expected_output_shape = (1000, )

if successful_requests:
    avg_latency = sum(r["latency_sec"] for r in successful_requests) / len(successful_requests)
    print(f"Average individual request latnecy (client-side): {avg_latency:.4f} seconds")

    correct_shapes = sum(1 for r in successful_requests if r["output_shape"] == expected_output_shape)
    print(f"Successfull requests: {len(successful_requests)} / {NUM_CONCURRENT_REQUESTS}")
    print(f"Requests with correct output shape {expected_output_shape}: {correct_shapes} / {len(successful_requests)}")
else:
    avg_latency = 0
    print("No successful requests.")
    correct_shapes = 0

print(f"Total time for {NUM_CONCURRENT_REQUESTS} requests: {total_time:.4f} seconds")
throughput = len(successful_requests) / total_time if total_time > 0 else 0
print(f"Approximate throughtput: {throughput:.2f} req/sec")
print("\nFinished tests for dynamic batching")