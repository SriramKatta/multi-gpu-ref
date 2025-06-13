#!/bin/bash -l

./compile.sh

sbatch --gres=gpu:a40:8 -J nccl_a40 --time=03:00:0 \
    ./nccl_bandwidth.sh ./executable/ncclsimplestream \
    "Stream+Ring Bechmark: Shift Bandwidth(GB/s) vs. message size" \
    "--lines "PCIe 4.0 interconnect bandwidth=63""

sbatch --gres=gpu:a40:8 -J nccl_a40_nocomp --time=03:00:0 \
    ./nccl_bandwidth.sh ./executable/ncclsimplestream_nocomp \
    "Ring Shift Bechmark: Bandwidth(GB/s) vs. message size" \
    "--lines "PCIe 4.0 interconnect bandwidth=63""

sbatch --gres=gpu:a100:8 -J nccl_a100_nocomp --time=0:59:0 \
    ./nccl_bandwidth.sh ./executable/ncclsimplestream_nocomp \
    "Ring Shift Bechmark: Bandwidth(GB/s) vs. message size" \
    "--lines "NVLINK interconnect bandwidth=600""

sbatch --gres=gpu:a100:8 -J nccl_a100 --time=0:59:0 \
    ./nccl_bandwidth.sh ./executable/ncclsimplestream \
    "Stream+Ring Bechmark: Shift Bandwidth(GB/s) vs. message size" \
    "--lines "NVLINK interconnect bandwidth=600""
