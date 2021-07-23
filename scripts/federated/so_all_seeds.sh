#!/bin/bash

#SBATCH --job-name=train
#SBATCH --comment="SO lr"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=0-23  # TODO: count
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
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
logdir="/checkpoint/pillutla/pfl/outputs_fed"
savedir="/checkpoint/pillutla/pfl/saved_models_fed"
common_args="--pfl_algo fedavg --dataset stackoverflow --train_batch_size 64 --eval_batch_size 1024 "
common_args="${common_args} --num_communication_rounds 2000 --num_clients_per_round 10 --num_local_epochs 1"
common_args="${common_args} --clip_grad_norm  --log_test_every_n_rounds 100 --max_num_clients_for_logging 1000"  # TODO!


# Populate the array
list_of_jobs=()

for seed in 1 2 3
do
for arch_size in tiny mini medium base
do
for lr in 1 10
do
    name="so_${arch_size}_lr${lr}_seed${seed}"
    task_params="--server_optimizer adam --server_lr 1e-2 --client_scheduler const \
        --client_optimizer sgd --client_lr ${lr} \
        --logfilename ${logdir}/${name} --savedir ${savedir}/${name} \
        --global_scheduler linear --global_warmup_fraction 0.1 \
        --arch_size ${arch_size} --seed ${seed} \
    "
    list_of_jobs+=("${task_params}")
done
done
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

time python -u train_pfl.py \
        ${common_args} \
        ${args}

echo "Job completed at $(date)"
