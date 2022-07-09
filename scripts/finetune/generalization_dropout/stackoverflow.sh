#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


#SBATCH --job-name=train
#SBATCH --comment="SO mini lr"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=0-3 # TODO: count
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
common_args="${common_args} --clip_grad_norm  --arch_size ${arch_size} --lr 0.1 "

# Populate the array
list_of_jobs=()

seed=1
state="stateful"
init="pretrained"
pfl_algo="fedalt"
train_mode="tr_layer"
layers_to_finetune="3"

for seed in 1
do
for dropout in 0.3 0.5 0.7 0.9
do
pretrained_name="so_${arch_size}_${train_mode}_${layers_to_finetune}_${pfl_algo}_do${dropout}_${init}_${state}_seed${seed}"
for ne in 5
do
    name="${pretrained_name}_ne${ne}"
    task_params=" --num_epochs_personalization  ${ne} \
        --personalized_dropout ${dropout} \
        --pretrained_model_path ${savedir}/${pretrained_name}/checkpoint.pt \
        --seed ${seed} --personalize_on_client ${train_mode} --layers_to_finetune ${layers_to_finetune}  \
        --logfilename ${logdir}/${name} --savedir ${savedir}/${name}"
    list_of_jobs+=("${task_params}")
done  # ne
done  # reg_param
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

# sleep 10m
# sleep ${job_id}m
sleep $(shuf -i 30-600 -n 1)  # sleep 0.5 to 10 mins

time python -u train_finetune.py \
        ${common_args} \
        ${args}

echo "Job completed at $(date)"
