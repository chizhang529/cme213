#!/bin/bash

#SBATCH --mem=20G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:4
#SBATCH --job-name=cme213
#SBATCH --output=cme213-%j.out
#SBATCH --error=cme213-%j.err

WORKDIR='/home/czhang94/project'
export WORKDIR

### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------

echo The master node of this job is `hostname`
echo This job runs on the following nodes:
echo `scontrol show hostname $SLURM_JOB_NODELIST`
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `echo $WORKDIR`"
echo
echo Output from code
echo ----------------

# mpirun -np 4 ./main -s -n 1000 -b 800 -l 0.01 -e 20
# echo "++--------------------++"
# echo "++      Case 1        ++"
# echo "++--------------------++"
# mpirun -np 4 ./main -g 1
# echo

# echo "++--------------------++"
# echo "++      Case 2        ++"
# echo "++--------------------++"
# mpirun -np 4 ./main -g 2
# echo

# echo "++--------------------++"
# echo "++      Case 3        ++"
# echo "++--------------------++"
# mpirun -np 4 ./main -g 3
# echo

echo "++--------------------++"
echo "++     Benchmark      ++"
echo "++--------------------++"
mpirun -np 4 ./main
echo

# echo "++--------------------++"
# echo "++      myGEMM        ++"
# echo "++--------------------++"
# mpirun -np 4 ./main -g 4 # myGEMM

# Profiling
# MV2_USE_CUDA=1 mpirun -np 4 nvprof --output-profile profile.%p.nvprof ./main -g 3
# MV2_USE_CUDA=1 mpirun -np 1 nvprof --kernels myGEMMKernel --analysis-metrics --output-profile GEMMmetrics.out.%p.nvprof ./main -g 1
