#!/bin/bash

#SBATCH --job-name=train
#SBATCH --comment="EMNIST tune l2 reg"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=0-23  # TODO: count
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --open-mode=append

source ~/.bashrc  # load all modules
source activate pyt19  # load environment

echo "Running [ ${0} ${@} ] on $(hostname), starting at $(date)"
echo "Job id = ${SLURM_JOB_ID}, task id = ${SLURM_ARRAY_TASK_ID}"
echo "PWD = $(pwd)"
echo "python path = $(which python)"

set -exu

# Default arguments
logdir="/checkpoint/pillutla/pfl/outputs2"
savedir="/checkpoint/pillutla/pfl/saved_models2"
common_args="--dataset emnist  --model_name resnet_gn --num_communication_rounds 2000 --train_batch_size 32 --eval_batch_size 256 "
common_args="${common_args} --num_clients_per_round 10 --num_local_epochs 1"
common_args="${common_args} --client_lr 0.5  --client_scheduler const --server_optimizer sgd --server_lr 1.0 --server_momentum 0.0 \
    --global_scheduler const_and_cut --global_lr_decay_factor 0.5 --global_lr_decay_every 500 \
    --validation_mode"

# Populate the array
list_of_jobs=()

for pfl_algo in "pfl_joint" "pfl_alternating"
do
for train_mode in "adapter"
do
for l2reg in 1e-1 1e-2 1e-3
do
for num_rounds in 500 2000
do
    pretrain_name="emnist_val_${pfl_algo}_${train_mode}_l2reg${l2reg}_init${num_rounds}"
    task_params=" --pfl_algo ${pfl_algo} \
        --logfilename ${logdir}/${name} --savedir ${savedir}/${name} \
        --pretrained_model_path /checkpoint/pillutla/pfl/saved_models2/${pretrain_name}/checkpoint.pt \
        --client_var_l2_reg_coef ${l2reg}  \
        "
    list_of_jobs+=("${task_params}")
done  # num_rounds
done  # l2reg
for iters in 2 4 8
do
for num_rounds in 500 2000
do
    pretrain_name="emnist_val_${pfl_algo}_${train_mode}_stop${iters}_init${num_rounds}"
    task_params=" --pfl_algo ${pfl_algo} \
        --logfilename ${logdir}/${name} --savedir ${savedir}/${name} \
        --pretrained_model_path /checkpoint/pillutla/pfl/saved_models2/${pretrain_name}/checkpoint.pt \
        --max_num_pfl_updates ${iters} \
        "
    list_of_jobs+=("${task_params}")
done  # num_rounds
done  # l2reg
done  # train mode
done  # pfl algo


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
