#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


#SBATCH --job-name=train
#SBATCH --comment="GLDv2 lr"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=0-5  # TODO: count
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
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

seed=1

# Populate the array
list_of_jobs=()

for state in "stateful"
do
for init in "pretrained"
do
for pfl_algo in "fedsim" "fedalt"
do
for train_mode in "inp_layer" "out_layer" "adapter"
do

pretrained_name="gldv2b_resnetgn_${train_mode}_${pfl_algo}_${init}_${state}_seed${seed}"
for ne in 5
do
    name="${pretrained_name}_ne${ne}"
    task_params="--num_epochs_personalization ${ne} \
        --pretrained_model_path ${savedir}/${pretrained_name}/checkpoint.pt \
        --seed ${seed} --personalize_on_client ${train_mode} \
        --logfilename ${logdir}/${name} --savedir ${savedir}/${name}"
    list_of_jobs+=("${task_params}")
done # ne
done # train_mode

done # pfl_algo
done # init
done # state

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
