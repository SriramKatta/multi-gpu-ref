#!/bin/bash -l
#
#SBATCH --output=./SLURM_OUT_FILES/%j_%x.out
#SBATCH --nodes=1
#SBATCH --time=3:0:00
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
benchmark with simple streaming kernl with cawre on $gpgpu version  and with count of $gpgpucount with exename $1
"

echo "$description"

module purge
module load openmpi/4.1.6-nvhpc23.7-cuda12
module load likwid


mainexe=$(realpath $1)
graphtitle="$2"
graphlimit="$3"
executable=./executable/cawaresimplestream_$SLURM_JOB_ID

cp $mainexe $executable


[ ! -d simdata ] && mkdir simdata

resfile=./simdata/${SLURM_JOB_ID}_caware

echo "#numproc(NP) message size(B) bandwidth(GB/s)" >"$resfile"

run_test() {
    for np in $(seq 2 $gpgpucount); do
    echo "$np of $gpgpucount"
        likwid-mpirun -np $np -nperdomain M:1 $executable | tee -a $resfile
    done
}

run_test

rm $executable

module load python
source ~/plotspace/bin/activate
python ../general_plotter.py "$resfile" "$graphtitle" "$graphlimit"
