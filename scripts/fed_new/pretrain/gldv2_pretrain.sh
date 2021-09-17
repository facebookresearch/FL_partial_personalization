#!/bin/bash

#SBATCH --job-name=train
#SBATCH --comment="GLDv2 lr"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=0-2 # TODO: count
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
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
common_args="--pfl_algo fedavg --dataset gldv2  --model_name resnet18 --train_batch_size 64 --eval_batch_size 128 "
common_args="${common_args} --num_clients_per_round 50 --num_local_epochs 1"
common_args="${common_args} --client_scheduler const --use_pretrained_model --log_test_every_n_rounds 150 \
            --client_lr 1e-2  --global_scheduler linear --server_optimizer adam --server_lr 2e-4  \
            "

# Populate the array
list_of_jobs=()

for num_rounds in 300 600 1200
do
    name="gldv2_pretrain_${num_rounds}"
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


time python -u train_pfl.py \
        ${common_args} \
        ${args}


echo "Job completed at $(date)"
