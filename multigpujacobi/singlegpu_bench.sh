#!/bin/bash -l
#
#SBATCH --output=./SLURM_OUT_FILES/%j_%x.out
#SBATCH --nodes=1
#SBATCH --time=0:50:00
#SBATCH --exclusive
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module purge
module load cuda nvhpc

[ ! -d simdata ] && mkdir simdata
resfile=./simdata/${SLURM_JOB_ID}_singlegpu

for i in {1..30}; do
    srun ./executable_perf/jacobi_single $(echo "1024*$i" | bc) | tee -a $resfile
done

nsys profile --stats=true -o ./simdata/${SLURM_JOB_ID}_jacobi_single ./executable_prof/jacobi_single 2048
