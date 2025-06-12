#!/bin/bash -l
#
#SBATCH --output=./SLURM_OUT_FILES/%j_%x.out
#SBATCH --nodes=1
#SBATCH --time=0:59:00
#SBATCH --exclusive
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

gpgpucount=$(nvidia-smi --query-gpu=gpu_name --format=csv | grep -i "nvidia" | wc -l)


module purge
module load cuda
module load likwid
module load openmpi/4.1.6-nvhpc23.7-cuda12

export NV_COMM_LIBS=$NVHPC_ROOT/Linux_x86_64/23.7/comm_libs/
#load NCCL library
export NCCL_HOME=$NV_COMM_LIBS/nccl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NCCL_HOME/lib

[ ! -d simdata ] && mkdir simdata
resfile=./simdata/${SLURM_JOB_ID}_caware

perfexemain="./executable_perf/jacobi_NCCL_overlap"
profexemain="./executable_prof/jacobi_NCCL_overlap"

perfexe="./executable_perf/jacobi_NCCL_overlap_${SLURM_JOB_ID}"
profexe="./executable_prof/jacobi_NCCL_overlap_${SLURM_JOB_ID}"

cp $perfexemain $perfexe
cp $profexemain $profexe

for np in $(seq 1 $gpgpucount); do
    echo "$np of $gpgpucount"
    likwid-mpirun -np $np -nperdomain M:1 $perfexe 40960 | tee -a $resfile
done

nsys profile --trace=mpi,cuda,nvtx --force-overwrite true \
    -o ./simdata/${SLURM_JOB_ID}_jacobi_NCCL_overlap \
    likwid-mpirun -np $gpgpucount -nperdomain M:1 \
    $profexe 4096

rm $profexe
rm $perfexe
