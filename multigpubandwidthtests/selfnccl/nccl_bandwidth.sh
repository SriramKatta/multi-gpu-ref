#!/bin/bash -l
#
#SBATCH --output=./SLURM_OUT_FILES/%j_%x.out
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --exclusive
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

gpgpucount=$(nvidia-smi --query-gpu=gpu_name --format=csv | grep -i "nvidia" | wc -l)
gpgpu=$(nvidia-smi --query-gpu=gpu_name \
    --format=csv |
    tail -n 1 |
    tr '-' ' ' |
    awk '{print $2}')

description="
benchmark with simple streaming kernl to compare with nccl on $gpgpu 80 gb version  and with count of $gpgpucount with exename $1
"

echo "$description"

module purge
module load likwid
module load cuda
module load openmpi/4.1.6-nvhpc23.7-cuda12

export NV_COMM_LIBS=$NVHPC_ROOT/Linux_x86_64/23.7/comm_libs/
#load NCCL library
export NCCL_HOME=$NV_COMM_LIBS/nccl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NCCL_HOME/lib

mainexe=$(realpath $1)
graphtitle="$2"
graphlimit="$3"
executable=./executable/ncclsimplestream_$SLURM_JOB_ID

cp $mainexe $executable


[ ! -d simdata ] && mkdir simdata

resfile=./simdata/${SLURM_JOB_ID}_nccl

echo -n >"$resfile"

run_test() {
    for np in $(seq 2 $gpgpucount); do
        echo "Running with $np GPUs" | tee -a $resfile
        likwid-mpirun -np $np -nperdomain M:1 $executable | tee -a $resfile
    done
}

run_test

rm $executable

module load python
source ~/plotspace/bin/activate
python ../general_plotter.py "$resfile" "$graphtitle" "$graphlimit"