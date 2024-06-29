from safetensors import safe_open

file_path = "/home/jzacharczuk/studia/nlp/chronos-forecasting/scripts/training/output/run-55/checkpoint-final/model.safetensors"

# Open the safetensors file
with safe_open(file_path, framework="pt") as f:
    # List all tensor keys in the file
    tensor_keys = f.keys()
    print("Tensors saved in the file:")
    for key in tensor_keys:
        print(key)
        tensor = f.get_tensor(key)
        print(f"Shape: {tensor.shape}, Dtype: {tensor.dtype}")
