#!/bin/bash -l

./compile.sh

sbatch --gres=gpu:a40:8 -J caware_a40_nocomp --time=02:30:0 \
    ./caware_bandwidth.sh ./executable/cawaresimplestream_nocomp \
    "Ring Shift Bechmark: Bandwidth(GB/s) vs. message size" \
    "--lines "PCIe 4.0 interconnect bandwidth=63""

sbatch --gres=gpu:a40:8 -J caware_a40 --time=02:30:0 \
    ./caware_bandwidth.sh ./executable/cawaresimplestream \
    "Stream+Ring Bechmark: Shift Bandwidth(GB/s) vs. message size" \
    "--lines  "NVLINK interconnect bandwidth=63" "Ring Shift Benchmark bandwidth(<5)=48" "Ring Shift Benchmark bandwidth(>4)=38""

sbatch --gres=gpu:a100:8 -J caware_a100_nocomp --time=0:59:0 \
    ./caware_bandwidth.sh ./executable/cawaresimplestream_nocomp \
    "Ring Shift Bechmark: Bandwidth(GB/s) vs. message size" \
    "--lines "NVLINK interconnect bandwidth=600""

sbatch --gres=gpu:a100:8 -J caware_a100 --time=0:59:0 \
    ./caware_bandwidth.sh ./executable/cawaresimplestream \
    "Stream+Ring Bechmark: Shift Bandwidth(GB/s) vs. message size" \
    "--lines  "NVLINK interconnect bandwidth=600" "Ring Shift Benchmark bandwidth=532""
