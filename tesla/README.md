## ECE Tesla Machine Distributed Training (Inter-Node)

This directory includes code of distributed training on Tesla machine (inter-node).


SETUP
=====

Set up passwordless SSH on 5 Tesla machines (ecetesla[0-4]). Default machines use `/bin/tsch` SHELL. First, change `WATID` to your watid and `NEWSHELL` to zsh. Then run

```sh
bash setup/change_cn_shell.sh
```

The script will also check for new updated shell.

Then change also `WATID` in `test_cn_connection.sh`, run the script, and make sure all 5 Tesla machines can retrieve the GPU information

```sh
bash setup/test_cn_connection.sh
```

You can also check their hostname, for example by this command

```sh
ssh ecetesla0 hostname
```

Put these hostnames into `setup/hosts.txt` and compile the first MPI simple hello world program on these 5 machines by

```sh
mpicc -o setup/hello_mpi setup/hello_mpi
```

Finally, we test OpenMPI by running

```sh
mpirun -np 5 --hostfile setup/hosts.txt .setup/hello_mpi
```

(Optional) Configure the .zshrc for more effective

We may use pip virtual environment and install `deepspeed`, `torch` (support CUDA 12.2 although nvidia-smi show CUDA Driver 12.4, otherwise error of pytorch is not compatible with RTX 3070's CUDA capability sm_86)

Native Nebula python (/usr/bin/python) (version 3.8.10) does not support python3-venv, we need to install non-pip virtual environment first, and manually install pip later

```sh
/usr/bin/python3.8 -m venv --without-pip vit_env
source vit_env/bin/activate
```

and install pip through

```sh
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
python -m pip install --upgrade pip
```

Verify pip version

```sh
pip --version
```

From [torch document](https://pytorch.org/get-started/locally/), install pytorch supporting CUDA 12.1 through

```
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify torch is working with all machines by addding `python3 /home/$WATID/Scalable_ViT_DT/tesla/script/test_torch_gpu.py` into entry script `entry_cn.sh`. This script will be run distributed using this command

```sh
mpirun -np 5 --hostfile setup/hosts.txt ./entry_cn.sh [YOUR_WATID]
```

Install necessary packages/libs through

```sh
python3 -m pip install -r requirement.txt
```

USING DEEPSPEED
===============

Create a `.deepspeed_env` file with the necessary environment variables. This file allows DeepSpeed to automatically propagate environment variables to each node.

```sh
echo "NCCL_IB_DISABLE=1" > ~/.deepspeed_env
echo "NCCL_SOCKET_IFNAME=^docker,lo" >> ~/.deepspeed_env
```

Install Deepspeed with MPI support using this command. And this can be found from [this](https://www.deepspeed.ai/getting-started/#multi-node-environment-variables)

```sh
DS_BUILD_OPS=1 DS_BUILD_AIO=0 DS_BUILD_KERNELS=1 DS_BUILD_MPU=1 DS_BUILD_MII=0 pip install deepspeed
``` 

Then run Deepspeed test with (enable `#TEST Deepspeed` option in ./entry_cn.sh)

```sh
mpirun -np 5 --hostfile setup/hosts.txt ./entry_cn.sh h3trinh > deepspeed/run_test_deepspeed.log
```

(THE CODE DID NOT WORK YET) Encounter this error, it seems we should not enable fp16 in DeepSpeed

```json
      "_comment": "fp16 is not supported in Tesla0 and Tesla3 ValueError: Type fp16 is not supported", 
      "fp16": {
        "enabled": false
      }
```

TODO
====

### Deepspeed

- [x] Initialize deepspeed and test it with dummy distributed matrix multiplcation on 5 machines (deepspeed/test_deepspeed_2.py)

- [x] Update the test to record GPU performance (using GPUtil)

- [x] Write Python script to plot/virtualize from this log

- [ ] Adapt the distributed training script

### Torch distributed

- [ ] Create torch distributed and run on 5 different machines to benchmarking the performance (baseline_measure.py)

- [ ] Update the pipeline to record these metrics

- [ ] Write Python script to plot/virtualize from this log

- [ ] Adapt the distributed training script
