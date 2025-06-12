#!/bin/bash -l
#
#SBATCH --output=./SLURM_OUT_FILES/%j_%x.out
#SBATCH --nodes=1
#SBATCH --time=3:0:00
#SBATCH --exclusive
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

gpgpucount=$(nvidia-smi --query-gpu=gpu_name --format=csv | grep -i "nvidia" | wc -l)

# module purge
# module load likwid cuda nvhpc openmpi

module purge
module load openmpi/4.1.6-nvhpc23.7-cuda12
module load likwid

[ ! -d simdata ] && mkdir simdata
resfile=./simdata/${SLURM_JOB_ID}_caware

perfexemain="./executable_perf/jacobi_caware"
profexemain="./executable_prof/jacobi_caware"

perfexe="./executable_perf/jacobi_caware_${SLURM_JOB_ID}"
profexe="./executable_prof/jacobi_caware_${SLURM_JOB_ID}"

cp $perfexemain $perfexe
cp $profexemain $profexe

for np in $(seq 1 $gpgpucount); do
    echo "$np of $gpgpucount"
    likwid-mpirun -np $np -nperdomain M:1 $perfexe 40960 | tee -a $resfile
done

likwid-mpirun -np 4 \
    nsys profile --trace=mpi,cuda,nvtx \
    -o ./simdata/${SLURM_JOB_ID}_jacobi_caware \
    $profexe 5120

rm $profexe
rm $perfexe
