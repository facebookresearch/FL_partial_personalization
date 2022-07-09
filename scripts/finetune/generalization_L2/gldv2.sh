#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


#SBATCH --job-name=train
#SBATCH --comment="GLDv2 lr"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=0-4  # TODO: count
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
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
common_args="${common_args} --lr 1e-3  "


# Populate the array
list_of_jobs=()

seed=1
state="stateful"
init="pretrained"
pfl_algo="fedalt"
train_mode="adapter"

for seed in 1
do
for reg_param in 1000 100 10 1 0.1
do
pretrained_name="gldv2b_resnetgn_${train_mode}_${pfl_algo}_reg${reg_param}_${init}_${state}_seed${seed}"
for ne in 5
do
    name="${pretrained_name}_ne${ne}"
    task_params="--num_epochs_personalization ${ne} \
        --client_var_l2_reg_coef ${reg_param} --client_var_prox_to_init \
        --pretrained_model_path ${savedir}/${pretrained_name}/checkpoint.pt \
        --seed ${seed} --personalize_on_client ${train_mode} \
        --logfilename ${logdir}/${name} --savedir ${savedir}/${name}"
    list_of_jobs+=("${task_params}")
done # ne
done # reg_param
done # seed


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
