#!/bin/bash

#SBATCH --job-name=train
#SBATCH --comment="GLDv2 lr"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=0-4  # TODO: count
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --open-mode=append

source ~/.bashrc  # load all modules
source activate pyt19  # load environment


echo "Running [ ${0} ${@} ] on $(hostname), starting at $(date)"
echo "Job id = ${SLURM_JOB_ID}, task id = ${SLURM_ARRAY_TASK_ID}"
echo "PWD = $(pwd)"
echo "python path = $(which python)"

set -exu

# Default arguments
logdir="/checkpoint/pillutla/pfl/outputs3"
savedir="/checkpoint/pillutla/pfl/saved_models3"
common_args="--dataset gldv2  --model_name resnet18 --train_batch_size 64 --eval_batch_size 128 "
common_args="${common_args} --lr 1e-3  \
        "

# Populate the array
list_of_jobs=()

l2reg=100
train_mode="finetune"


for ne in 5
do
for seed in 1 2 3 4 5
do
    pretrained_name="gldv2b_resnetgn_pfedme_seed${seed}"
    name="${pretrained_name}_ne${ne}"
    task_params="--num_epochs_personalization ${ne} \
        --pretrained_model_path ${savedir}/${pretrained_name}/checkpoint.pt \
        --client_var_l2_reg_coef ${l2reg}  --client_var_prox_to_init \
        --seed ${seed} --personalize_on_client ${train_mode} \
        --logfilename ${logdir}/${name} --savedir ${savedir}/${name}"
    list_of_jobs+=("${task_params}")
done # seed
done # ne



# Run
num_jobs=${#list_of_jobs[@]}
job_id=${SLURM_ARRAY_TASK_ID}
if [ ${job_id} -ge ${num_jobs} ] ; then
    echo "Invalid job id; qutting"
    exit 2
fi
echo "-------- STARTING JOB ${job_id}/${num_jobs}"
args=${list_of_jobs[${job_id}]}

time python -u train_finetune.py \
        ${common_args} \
        ${args}

echo "Job completed at $(date)"