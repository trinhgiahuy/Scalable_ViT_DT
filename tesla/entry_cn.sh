#!/bin/bash
WATID=$1

# Replace this to source new virtual environment
source /home/$WATID/vit_env/bin/activate

python3 /home/$WATID/Scalable_ViT_DT/tesla/script/test_torch_gpu.py
