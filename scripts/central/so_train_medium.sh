#!/bin/bash

#SBATCH --job-name=train
#SBATCH --comment="Train SO"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=45:00:00
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

name="so_medium_try1"
save_params=" --logfilename ${logdir}/${name} --savefilename ${savedir}/${name}.pt"
arch_params="--arch_size medium"
log_params="\
            --max_num_clients_for_logging 100 \
            --log_train_every_n_clients 1000 \
            --log_test_every_n_clients 5000"
train_params="\
            --train_batch_size 64  \
            --eval_batch_size 1024 \
            --central_optimizer adam \
            --use_warmup \
            --num_warmup_updates 5000 \
            --num_epochs_centralized 2 \
        "

time python -u train_centralized.py \
            --dataset stackoverflow \
            ${train_params}  \
            ${log_params}  \
            ${arch_params}  \
            ${save_params} 

echo "Job completed at $(date)"
