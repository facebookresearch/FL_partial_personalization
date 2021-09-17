#!/bin/bash

#SBATCH --job-name=train
#SBATCH --comment="SO tiny lr"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=0-9  # TODO: count
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

arch_size="tiny"

# Default arguments
logdir="/checkpoint/pillutla/pfl/outputs3"
savedir="/checkpoint/pillutla/pfl/saved_models3"
common_args=" --dataset stackoverflow --train_batch_size 64 --eval_batch_size 1024 "
common_args="${common_args} --clip_grad_norm  --arch_size ${arch_size} --lr 1 \
    "

# Populate the array
list_of_jobs=()

l2reg=0.01
train_mode="finetune"

declare -A max_num_updates=( [1]=10000 [3]=3 )

for ne in 3 1 
do
for seed in 1 2 3 4 5
do
    pretrained_name="so_${arch_size}_pfedme_seed${seed}"
    name="${pretrained_name}_ne${ne}"
    task_params=" --num_epochs_personalization  ${ne} \
        --max_num_finetune_updates ${max_num_updates[${ne}]}
        --pretrained_model_path ${savedir}/${pretrained_name}/checkpoint.pt \
        --client_var_l2_reg_coef ${l2reg}  --client_var_prox_to_init \
        --seed ${seed} --personalize_on_client ${train_mode} \
        --logfilename ${logdir}/${name} --savedir ${savedir}/${name}"
    list_of_jobs+=("${task_params}")
done # seed
done  # ne


# Run
num_jobs=${#list_of_jobs[@]}
job_id=${SLURM_ARRAY_TASK_ID}
if [ ${job_id} -ge ${num_jobs} ] ; then
    echo "Invalid job id; qutting"
    exit 2
fi
echo "-------- STARTING JOB ${job_id}/${num_jobs}"
args=${list_of_jobs[${job_id}]}

sleep 10m

sleep ${job_id}m

time python -u train_finetune.py \
        ${common_args} \
        ${args}

echo "Job completed at $(date)"
