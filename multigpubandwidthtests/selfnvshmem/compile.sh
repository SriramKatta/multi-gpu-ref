#!/bin/bash -l

module purge
module load cmake

module load openmpi/4.1.6-nvhpc23.7-cuda12
export NV_COMM_LIBS=$NVHPC_ROOT/Linux_x86_64/23.7/comm_libs

#load nccl library 
export NCCL_HOME=$NV_COMM_LIBS/nccl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NCCL_HOME/lib
#load nvshmem library 
export NVSHMEM_HOME=$NV_COMM_LIBS/nvshmem
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVSHMEM_HOME/lib


cmake -S . -B build -DNVSHMEM_HOME=$NVSHMEM_HOME
cmake --build build -j

