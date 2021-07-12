#!/bin/bash

#SBATCH --job-name=train
#SBATCH --comment="EMNIST SO"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600

source ~/.bashrc  # load all modules
source activate pyt19  # load environment

echo "Running [ ${0} ${@} ] on $(hostname), starting at $(date)"
echo "Job id = ${SLURM_JOB_ID}, task id = ${SLURM_ARRAY_TASK_ID}"
echo "PWD = $(pwd)"
echo "python path = $(which python)"

set -exu

# Default arguments
logdir="/checkpoint/pillutla/pfl/outputs"
savedir="/checkpoint/pillutla/pfl/saved_models"
common_args="--dataset emnist  --model_name resnet --train_batch_size 128 --eval_batch_size 256 "
common_args="${common_args} --num_epochs 20"

name="emnist_resnet"
task_params=" --logfilename ${logdir}/${name} --savefilename ${savedir}/${name}.pt"
task_params="${task_params} --lr 1e-2 --lr_decay_factor 2 --lr_decay_every 3342"


time python -u train_centralized.py \
        ${common_args} \
        ${task_params}

echo "Job completed at $(date)"
