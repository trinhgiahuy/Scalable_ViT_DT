# Result Accumulation

For mixed-precision FP16 training, we tested on 3 RTX3070 GPUs only. We increased to `micro_batch_per_gpu` to 32 which is maximum supported on 3 GPUs.

![strong_scaling_fp16.png](strong_scaling_fp16.png)

For full floating point training, We use `micro_batch_per_gpu` to 16 which is maximum supported on 5 GPUs.
