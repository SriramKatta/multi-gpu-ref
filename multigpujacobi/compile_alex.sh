#!/bin/bash -l

module purge
module load openmpi/4.1.6-nvhpc23.7-cuda12
#module load hpcx
module load cmake

#export NVHPC_ROOT=/apps/SPACK/0.22.2/opt/linux-almalinux8-zen/gcc-8.4.1/nvhpc-25.5-5tjelrgdrvojmmo2k3qbvnusezsijacy
# export NV_COMM_LIBS=$NVHPC_ROOT/Linux_x86_64/23.7/comm_libs/
# #export NV_COMM_LIBS=/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-8.5.0/nvhpc-23.7-bzxcokzjvx4stynglo4u2ffpljajzlam/Linux_x86_64/23.7/comm_libs

# #load nvshmem library 
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVSHMEM_HOME/lib
# export NVSHMEM_HOME=$NV_COMM_LIBS/nvshmem

export NV_COMM_LIBS=$NVHPC_ROOT/Linux_x86_64/23.7/comm_libs/
#load nvshmem library 
export NCCL_HOME=$NV_COMM_LIBS/nccl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NCCL_HOME/lib
#load nccl library 
export NCCL_HOME=$NV_COMM_LIBS/nccl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NCCL_HOME/lib



cmake -S . -B build -DNCCL_HOME=$NCCL_HOME -DNVSHMEM_HOME=$NVSHMEM_HOME
cmake --build build -j

cmake -S . -B build -DNCCL_HOME=$NCCL_HOME -DNVSHMEM_HOME=$NVSHMEM_HOME -DNVTX_OFF=OFF
cmake --build build -j

