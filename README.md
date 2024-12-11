# ViT-distributed-training

The repository source code for research project "Scaling Vision Transformers: Evaluating DeepSpeed for Image-Centric Workloads"

The directory includes both intra-node (on Nebula machine) and inter-node (on Tesla machine) distributed training follow data parallelism paradigm for now. The directory will benchmark the scalability (weak/strong scaling) and in detail report the configuration of required system (memory, structure). 

See `tesla`, `nebula` and `vector` folders for more information.
