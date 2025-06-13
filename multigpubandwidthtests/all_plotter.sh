#!/bin/bash -l

module load python
source ~/plotspace/bin/activate

#caware
#a40
python ./general_plotter.py ./selfcawarempi/simdata/2738671_caware "Ring Shift Benchmark: Bandwidth(GB/s) vs. Message Size" --lines "PCIe 4.0 interconnect bandwidth=63"
python ./general_plotter.py ./selfcawarempi/simdata/2738670_caware "Stream+Ring Benchmark: Bandwidth(GB/s) vs. Message Size" --lines "PCIe 4.0 interconnect bandwidth=63" "Ring Shift Benchmark bandwidth(>4)=41" "Ring Shift Benchmark bandwidth(<4)=52"
#a100
python ./general_plotter.py ./selfcawarempi/simdata/2738672_caware "Ring Shift Benchmark: Bandwidth(GB/s) vs. Message Size" --lines "NVlink interconnect bandwidth=600"
python ./general_plotter.py ./selfcawarempi/simdata/2738673_caware "Stream+Ring Benchmark: Bandwidth(GB/s) vs. Message Size" --lines "NVlink interconnect bandwidth=600" "Ring Shift Benchmark bandwidth=532"

#nccl
#a40
python ./general_plotter.py ./selfnccl/simdata/2738675_nccl "Ring Shift Benchmark: Bandwidth(GB/s) vs. Message Size" --lines "PCIe 4.0 interconnect bandwidth=63"
python ./general_plotter.py ./selfnccl/simdata/2738674_nccl "Stream+Ring Benchmark: Bandwidth(GB/s) vs. Message Size" --lines "PCIe 4.0 interconnect bandwidth=63" "Ring Shift Benchmark bandwidth(>4)=32" "Ring Shift Benchmark bandwidth(<4)=46"
#a100
python ./general_plotter.py ./selfnccl/simdata/2738676_nccl "Ring Shift Benchmark: Bandwidth(GB/s) vs. Message Size" --lines "NVlink interconnect bandwidth=600"
python ./general_plotter.py ./selfnccl/simdata/2738677_nccl "Stream+Ring Benchmark: Bandwidth(GB/s) vs. Message Size" --lines "NVlink interconnect bandwidth=600" "Ring Shift Benchmark bandwidth=366"

#nvshemem
#a40
python ./general_plotter.py ./selfnvshmem/simdata/2738754_nvs "Ring Shift Benchmark: Bandwidth(GB/s) vs. Message Size" --lines "PCIe 4.0 interconnect bandwidth=63"
python ./general_plotter.py ./selfnvshmem/simdata/2738757_nvs "Stream+Ring Benchmark: Bandwidth(GB/s) vs. Message Size" --lines "PCIe 4.0 interconnect bandwidth=63" "Ring Shift Benchmark bandwidth(>4)=28" "Ring Shift Benchmark bandwidth(<4)=40"
#a100
python ./general_plotter.py ./selfnvshmem/simdata/2738755_nvs "Ring Shift Benchmark: Bandwidth(GB/s) vs. Message Size" --lines "NVlink interconnect bandwidth=600"
python ./general_plotter.py ./selfnvshmem/simdata/2738756_nvs "Stream+Ring Benchmark: Bandwidth(GB/s) vs. Message Size" --lines "NVlink interconnect bandwidth=600" "Ring Shift Benchmark bandwidth=441"
