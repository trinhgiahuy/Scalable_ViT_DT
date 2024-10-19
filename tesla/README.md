## ECE Tesla Machine Distributed Training

This directory includes code of distributed training on Tesla machine.

### SETUP

Set up passwordless SSH on 5 Tesla machines (ecetesla[0-4]). Default machines use `/bin/tsch` SHELL. First, change `WATID` to your watid and `NEWSHELL` to zsh. Then run

```sh
bash setup/change_shell_cn.sh
```

The script will also check for new updated shell.

Then run `test_cn_connection.sh` and make sure all 5 Tesla machines can retrieve the GPU information

```sh
bash setup/test_cn_connection.sh`
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

### TODO
