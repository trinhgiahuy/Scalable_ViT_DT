# ViT-distributed-training

The directory includes both intra-node (on Nebula machine) and inter-node (on Tesla machine) distributed training follow data parallelism paradigm for now. The directory will benchmark the scalability (weak/strong scaling) and in detail report the configuration of required system (memory, structure). 

Install
=======

```sh
git clone git@github.com:trinhgiahuy/Scalable_ViT_DT.git
```

For Tesla

```sh
cd tesla
python3 -m pip install -r requirement.txt
```

