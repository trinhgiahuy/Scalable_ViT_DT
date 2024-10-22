#!/bin/bash
WATID=$1

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# ===================Set NCCL environment variables to force TCP sockets
# Disables InfiniBand if it's not available on the network. It forces NCCL to use TCP over sockets
export NCCL_IB_DISABLE=1                  
# Ensures that NCCL does not attempt to use GDR (GPUDirect RDMA) if not supported.
export NCCL_NET_GDR_LEVEL=0        
# Avoid using Docker and loopback interfaces        
export NCCL_SOCKET_IFNAME=^docker,lo       
# [WILL THIS DEGRADE PERFORMANCE]? Set NCCL protocol to simple to simplify communication
export NCCL_PROTO=simple                   
# ===================



# Make sure CUDA_VISIBLE_DEVICES is set properly to avoid device ordinal 
# Since there is only 1 GPU per 1 node
export CUDA_VISIBLE_DEVICES=0              

# Replace this to source new virtual environment
source /home/$WATID/vit_env/bin/activate

# Uncomment this to test torch accross distributed GPU
# python3 /home/$WATID/Scalable_ViT_DT/tesla/script/test_torch_gpu.py

# Uncomment this to test the NCCL between tesla0 and tesla1
# python3 /home/$WATID/Scalable_ViT_DT/tesla/simple_test/test_nccl.py

# Uncomment this to start running dummy example
python3 /home/$WATID/Scalable_ViT_DT/tesla/script/baseline_measure.py