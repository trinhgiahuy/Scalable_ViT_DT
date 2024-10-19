#!/bin/bash

watid=h3trinh
# List of Tesla nodes
nodes=("ecetesla0" "ecetesla1" "ecetesla2" "ecetesla3" "ecetesla4")

# Set a timeout in seconds
SSH_TIMEOUT=10

# Loop through each node and test SSH
for host in "${nodes[@]}"; do
    echo "Testing passwordless SSH to $host..."

    # Test if passwordless SSH is working with a timeout
    #timeout $SSH_TIMEOUT ssh -o BatchMode=yes $host "echo 'Passwordless SSH to $(hostname) successful'"

    #if [ $? -ne 0 ]; then
    #    echo "SSH to $host failed or timed out"
    #    continue
    #fi

    # Check NVCC version
    ssh $watid@$host "nvcc --version"

    # Check NVIDIA GPU status
    # timeout $SSH_TIMEOUT ssh -o BatchMode=yes $host "nvidia-smi"
    ssh $watid@$host "nvidia-smi"

    echo "Test on $host completed."
    echo "------------------------------------"
done

