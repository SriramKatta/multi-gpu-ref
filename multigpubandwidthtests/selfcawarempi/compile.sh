#!/bin/bash -l

module purge
module load openmpi/4.1.6-nvhpc23.7-cuda12
module load cmake


cmake -S . -B build 
cmake --build build -j

