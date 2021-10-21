#!/bin/bash

#SBATCH --job-name=train
#SBATCH --comment="SO mini lr"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=0-5  # TODO: count
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=3:30:00
#SBATCH --open-mode=append

source ~/.bashrc  # load all modules
source activate pyt19  # load environment

echo "Running [ ${0} ${@} ] on $(hostname), starting at $(date)"
echo "Job id = ${SLURM_JOB_ID}, task id = ${SLURM_ARRAY_TASK_ID}"
echo "PWD = $(pwd)"
echo "python path = $(which python)"

set -exu

arch_size="mini"

# Default arguments
logdir="/checkpoint/pillutla/pfl/outputs3"
savedir="/checkpoint/pillutla/pfl/saved_models3"
common_args=" --dataset stackoverflow --train_batch_size 64 --eval_batch_size 1024 "
common_args="${common_args} --clip_grad_norm  --arch_size ${arch_size} \
        --pretrained_model_path /checkpoint/pillutla/pfl/saved_models2/so_${arch_size}_pretrain_val/checkpoint.pt \
        --validation_mode"

# Populate the array
list_of_jobs=()

seed=0

for lr in 1 0.1 0.01
do
for ne in 1 5 
do
train_mode="finetune"
    name="so_${arch_size}_fedavg_finetune_lr${lr}_ne${ne}"
    task_params="--lr ${lr} --num_epochs_personalization  ${ne} \
        --seed ${seed} --personalize_on_client ${train_mode} \
        --logfilename ${logdir}/${name} --savedir ${savedir}/${name}"
    list_of_jobs+=("${task_params}")
done  # ne
done  # lr


# Run
num_jobs=${#list_of_jobs[@]}
job_id=${SLURM_ARRAY_TASK_ID}
if [ ${job_id} -ge ${num_jobs} ] ; then
    echo "Invalid job id; qutting"
    exit 2
fi
echo "-------- STARTING JOB ${job_id}/${num_jobs}"
args=${list_of_jobs[${job_id}]}

# sleep 10m

sleep ${job_id}m

time python -u train_finetune.py \
        ${common_args} \
        ${args}

echo "Job completed at $(date)"