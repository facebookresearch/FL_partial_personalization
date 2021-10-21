#!/bin/bash

#SBATCH --job-name=train
#SBATCH --comment="EMNIST main"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=0-5  # TODO: count
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=6:30:00
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
common_args="--dataset emnist  --model_name resnet_gn --train_batch_size 32 --eval_batch_size 256 "
common_args="${common_args} --num_communication_rounds 500 --num_clients_per_round 10 --num_local_epochs 1"
common_args="${common_args}  --client_scheduler const --server_optimizer sgd --server_lr 1.0 --server_momentum 0.0 \
    --client_lr 0.01 --global_scheduler const_and_cut --global_lr_decay_factor 0.5 --global_lr_decay_every 500 \
        --pretrained_model_path /checkpoint/pillutla/pfl/saved_models2/emnist_pretrain_2000/checkpoint.pt \
    "

seed=1

# Populate the array
list_of_jobs=()

for pfl_algo in "pfl_joint" "pfl_alternating"
do
for train_mode in "inp_layer" "out_layer" "adapter"
do
    name="emnist_resnetgn_${train_mode}_${pfl_algo}_pretrained_stateful_seed${seed}"
    task_params="--seed ${seed} --pfl_algo ${pfl_algo} --personalize_on_client ${train_mode} --logfilename ${logdir}/${name} --savedir ${savedir}/${name}"
    list_of_jobs+=("${task_params}")
done # train_mode
done  # pfl_algo


# Run
num_jobs=${#list_of_jobs[@]}
job_id=${SLURM_ARRAY_TASK_ID}
if [ ${job_id} -ge ${num_jobs} ] ; then
    echo "Invalid job id; qutting"
    exit 2
fi
echo "-------- STARTING JOB ${job_id}/${num_jobs}"
args=${list_of_jobs[${job_id}]}

time python -u train_pfl.py \
        ${common_args} \
        ${args}

echo "Job completed at $(date)"
