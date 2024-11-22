import deepspeed
print("DeepSpeed version:", deepspeed.__version__)

import torch
print("Pytorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("Is CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        device_props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Total Memory: {device_props.total_memory / 1e9} GB")

        # Check if memory_clock_rate attribute exists
        if hasattr(device_props, 'memory_clock_rate'):
            print(f"  Memory Clock Rate: {device_props.memory_clock_rate / 1e6} GHz")
        else:
            print("  Memory Clock Rate: Not available")

        print(f"  Multiprocessor Count: {device_props.multi_processor_count}")
        print(f"  CUDA Cores: {device_props.multi_processor_count * 64}")  # Approximation
else:
    print("No GPU found")

