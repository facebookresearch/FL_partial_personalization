#!/bin/bash

#SBATCH --job-name=train
#SBATCH --comment="SO tiny main"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=3  # TODO: count
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=14:30:00
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
common_args="${common_args} --num_communication_rounds 500 --num_clients_per_round 10 --num_local_epochs 1"
common_args="${common_args} --clip_grad_norm  --log_test_every_n_rounds 50 --max_num_clients_for_logging 1000 "  # TODO!
common_args="${common_args} --arch_size ${arch_size} \
        --server_optimizer adam --server_lr 1e-3 --client_lr 1 --client_scheduler const \
        --client_optimizer sgd --global_scheduler linear --global_warmup_fraction 0.1 \
        --pretrained_model_path /checkpoint/pillutla/pfl/saved_models2/so_tiny_pretrain_val/checkpoint.pt \
        --validation_mode \
        "

# Populate the array
list_of_jobs=()
seed=0

for l2reg in 0.1 0.01 0.001 0.0001 1e-5
do
    name="so_${arch_size}_pfedme_l2reg${l2reg}_val"
    task_params="--seed ${seed} --pfl_algo pfedme --pfedme_l2_reg_coef ${l2reg} --logfilename ${logdir}/${name} --savedir ${savedir}/${name}"
    list_of_jobs+=("${task_params}")
done  # l2reg


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

time python -u train_pfl.py \
        ${common_args} \
        ${args}

echo "Job completed at $(date)"
