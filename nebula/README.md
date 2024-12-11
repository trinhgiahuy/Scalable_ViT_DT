### Intra-node ECE Tesla Machine Distributed Training

This directory includes code of distributed training on Nebula Computing Cluster (Intra-node)

## SETUP

Create python virtual environment by `python3 -m venv vit_env` then `source vit_env/bin/activate`

Install packages through

```sh
pip install -r requirement.txt
```

Submit the training script through

```sh
sbatch submit_distributed_vit_training.sh
```
