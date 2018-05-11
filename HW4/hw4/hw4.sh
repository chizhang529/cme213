#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:4
#SBATCH --job-name=cme213
#SBATCH --output=cme213-%j.out
#SBATCH --error=cme213-%j.err

echo "In file hw4.sh, update the line below before running the script"
echo "WORKDIR='<directory with your code>'"
exit 0

# Comment the 3 lines above after setting WORKDIR

WORKDIR='<directory with your code>'
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

########### UPDATE THESE VARIABLES ###############
cipher_period=11
##################################################

# Encrypt text
./create_cipher mobydick.txt $cipher_period
# Decipher text
./solve_cipher cipher_text.txt
