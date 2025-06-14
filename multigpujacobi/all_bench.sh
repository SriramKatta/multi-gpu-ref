#!/bin/bash -l

./compile_alex.sh

for gpu in "a40" "a100"; do
    for sscript in "./singlegpu_bench.sh" "./caware_bench.sh" "./caware_overlap_bench.sh" "./nccl_overlap_bench.sh" "./nccl_overlap_graph_bench.sh" "./nvs_bench.sh"; do
        echo "dispatch $sscript on GPU $gpu"
        sbatch --gres=gpu:$gpu:8 $sscript
    done
done
