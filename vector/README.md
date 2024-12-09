### Intra-node ECE Tesla Machine Distributed Training

Commands to setup the environment on Vector Cluster


```sh
mkdir $HOME/condaenvs
export PATH=/pkgs/anaconda3/bin:$PATH
which conda
conda create -p ~/vit_env python=3.9 -y
conda init bash

export CUDA_HOME=/pkgs/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

pip install deepspeed
pip install tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install deepspeed
pip install gputil
CC=mpicc pip install --no-cache-dir mpi4py
```

Submit intra-node training with
```sh
sbatch submit_intra_node.sh
```

Submit inter-node training with
```sh
sbatch submit_inter-node.sh
```

