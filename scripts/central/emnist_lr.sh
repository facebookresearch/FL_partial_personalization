#!/bin/bash

#SBATCH --job-name=train
#SBATCH --comment="EMNIST SO"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=0-19  # TODO: count
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

# Populate the array
list_of_jobs=()
for lr in 2e-2 1e-2 5e-3 2.5e-3 1e-3
do
# Use rounds (epoch=multiples of number of train clients) to decay
for lre in 1114 2228 3342 4456
do
    name="emnist_resnet_lr${lr}_lre${lre}"
    task_params=" --logfilename ${logdir}/${name} --savefilename ${savedir}/${name}.pt"
    task_params="${task_params} --lr ${lr} --lr_decay_factor 2 --lr_decay_every ${lre}"
    list_of_jobs+=("${task_params}")
done
done


# Run
num_jobs=${#list_of_jobs[@]}
job_id=${SLURM_ARRAY_TASK_ID}
if [ ${job_id} -ge ${num_jobs} ] ; then
    echo "Invalid job id; qutting"
    exit 2
fi
echo "-------- STARTING JOB ${job_id}/${num_jobs}"
args=${list_of_jobs[${job_id}]}

time python -u train_centralized.py \
        ${common_args} \
        ${args}

echo "Job completed at $(date)"
