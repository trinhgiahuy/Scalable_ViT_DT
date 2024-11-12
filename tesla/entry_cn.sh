#!/bin/bash
WATID=$1
NCCL_DEBUG_OPT=0

if [ "$NCCL_DEBUG_OPT" -eq 1 ]; then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=ALL
    echo "NCCL Debugging is enabled"
else
    # export NCCL_DEBUG=OFF
    unset NCCL_DEBUG
    unset NCCL_DEBUG_SUBSYS
    echo "NCCL Debugging is disabled."
fi

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
# Set CUDA_VISIBLE_DEVICES to map GPUs across nodes
if [ "$HOSTNAME" == "ecetesla0" ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ "$HOSTNAME" == "ecetesla1" ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ "$HOSTNAME" == "ecetesla2" ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ "$HOSTNAME" == "ecetesla3" ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ "$HOSTNAME" == "ecetesla4" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi


# Replace this to source new virtual environment
source /home/$WATID/vit_env/bin/activate

# Uncomment this to test torch accross distributed GPU
# python3 /home/$WATID/Scalable_ViT_DT/tesla/script/test_torch_gpu.py

# Uncomment this to test the NCCL between tesla0 and tesla1
# python3 /home/$WATID/Scalable_ViT_DT/tesla/simple_test/test_nccl.py

# Uncomment this to start running dummy example
# python3 /home/$WATID/Scalable_ViT_DT/tesla/script/baseline_measure.py

#TEST Scalene
# scalene --gpu --reduced-profile --cpu --profile-interval 1 --outfile metrics_scalene.log python3 /home/$WATID/Scalable_ViT_DT/tesla/script/test_scalene.py

#TEST Deepspeed
# deepspeed --launcher mpirun --hostfile /home/$WATID/Scalable_ViT_DT/tesla/setup/hosts.txt /home/$WATID/Scalable_ViT_DT/tesla/deepspeed/test_deepspeed.py
# python3 /home/$WATID/Scalable_ViT_DT/tesla/deepspeed/test_deepspeed_2.py

#TEST test distributed training
python3 /home/$WATID/Scalable_ViT_DT/tesla/deepspeed/run_distributed_training.py --deepspeed_config "deepspeed_config.json"