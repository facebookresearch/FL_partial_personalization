#!/bin/bash

#SBATCH --job-name=train
#SBATCH --comment="Finetune SO"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=0-6  # TODO: count
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
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
modelfilename="/checkpoint/pillutla/pfl/saved_models/so_tiny_try1.pt"

arch_params="\
            --num_attn_heads 2 \
            --num_transformer_layers 2 \
            --input_dim 128 \
            --attn_hidden_dim 64 \
            --fc_hidden_dim 512 \
            --dropout_tr 0 \
            --dropout_io 0 \
            "
train_params="\
            --train_batch_size 32 \
            --eval_batch_size 1024 \
            --optimizer adam \
            --scheduler linear \
            --lr 3.5e-4 \
        "
common_params="\
            --num_updates_personalization 100 \
            --modelfilename ${modelfilename} \
        "

list_of_jobs=()

train_mode="adapter"
for hidden_dim in 2 4 8 16 32 64 128
do
    name="so_tiny_try1_${train_mode}_${hidden_dim}"
    task_params="--train_mode ${train_mode} --adapter_hidden_dim ${hidden_dim} --logfilename ${logdir}/${name}"
    list_of_jobs+=("${task_params}")
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

time python -u train_personalized_finetune.py \
            --dataset stackoverflow \
            ${train_params}  \
            ${common_params}  \
            ${arch_params}  \
            ${args} 

echo "Job completed at $(date)"