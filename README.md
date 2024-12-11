# ViT-distributed-training

The repository source code for research project "caling Vision Transformers: Evaluating DeepSpeed for Image-Centric Workloads"

The directory includes both intra-node (on Nebula machine) and inter-node (on Tesla machine) distributed training follow data parallelism paradigm for now. The directory will benchmark the scalability (weak/strong scaling) and in detail report the configuration of required system (memory, structure). 

Install
=======

```sh
git clone git@github.com:trinhgiahuy/Scalable_ViT_DT.git
```

See `Tesla`, `Nebula` and `Vector` folders for more information.
