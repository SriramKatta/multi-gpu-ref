#!/bin/bash -l

./compile.sh

sbatch --gres=gpu:a40:8 -J nvs_a40_nocomp --time=04:00:0 \
    ./nvs_bandwidth.sh ./executable/nvshmemsimpestream_nocomp \
    "Ring Shift Bechmark: Bandwidth(GB/s) vs. message size" \
    "--lines "PCIe 4.0 interconnect bandwidth=63""

sbatch --gres=gpu:a40:8 -J nvs_a40 --time=04:00:0 \
    ./nvs_bandwidth.sh ./executable/nvshmemsimpestream \
    "Stream+Ring Bechmark: Shift Bandwidth(GB/s) vs. message size" \
    "--lines  "NVLINK interconnect bandwidth=63" "Ring Shift Benchmark bandwidth=48" "Ring Shift Benchmark bandwidth=38""

sbatch --gres=gpu:a100:8 -J nvs_a100_nocomp --time=0:59:0 \
    ./nvs_bandwidth.sh ./executable/nvshmemsimpestream_nocomp \
    "Ring Shift Bechmark: Bandwidth(GB/s) vs. message size" \
    "--lines "NVLINK interconnect bandwidth=600""

sbatch --gres=gpu:a100:8 -J nvs_a100 --time=0:59:0 \
    ./nvs_bandwidth.sh ./executable/nvshmemsimpestream \
    "Stream+Ring Bechmark: Shift Bandwidth(GB/s) vs. message size" \
    "--lines "NVLINK interconnect bandwidth=600" "Ring Shift Benchmark bandwidth=442""
