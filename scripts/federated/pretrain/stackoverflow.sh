#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


#SBATCH --job-name=train
#SBATCH --comment="SO lr"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=0-3  # TODO: count
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
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
logdir="/checkpoint/pillutla/pfl/outputs2"
savedir="/checkpoint/pillutla/pfl/saved_models2"
common_args=" --dataset stackoverflow --train_batch_size 64 --eval_batch_size 1024 "
common_args="${common_args} --num_clients_per_round 50 --num_local_epochs 1"
common_args="${common_args} --clip_grad_norm  --log_test_every_n_rounds 100 --max_num_clients_for_logging 1000 "  # TODO!
common_args="${common_args} --arch_size ${arch_size} --pfl_algo fedavg \
        --server_optimizer adam --server_lr 5e-4 --client_scheduler const --client_lr 1 \
        --client_optimizer sgd --global_scheduler linear --global_warmup_fraction 0.1 \
        "

# Populate the array
list_of_jobs=()


for num_rounds in 1000
do
    name="so_${arch_size}_pretrain_${num_rounds}"
    task_params="--num_communication_rounds ${num_rounds} --logfilename ${logdir}/${name} --savedir ${savedir}/${name}"
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

sleep ${job_id}m

time python -u train_pfl.py \
        ${common_args} \
        ${args}

echo "Job completed at $(date)"
