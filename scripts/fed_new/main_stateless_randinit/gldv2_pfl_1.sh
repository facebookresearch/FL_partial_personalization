#!/bin/bash

#SBATCH --job-name=train
#SBATCH --comment="EMNIST lr"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=0-1  # TODO: count
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
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
logdir="/checkpoint/pillutla/pfl/outputs3"
savedir="/checkpoint/pillutla/pfl/saved_models3"
common_args="--dataset gldv2  --model_name resnet18 --train_batch_size 64 --eval_batch_size 128 "
common_args="${common_args} --num_communication_rounds 2500 --num_clients_per_round 50 --num_local_epochs 1"
common_args="${common_args} --client_scheduler const --use_pretrained_model --log_test_every_n_rounds 150 \
            --global_scheduler linear --server_optimizer adam --server_lr 2e-4 --client_lr 1e-2  \
            --stateless_clients"

# Populate the array
list_of_jobs=()

seed=1

for pfl_algo in "pfl_joint" "pfl_alternating"
do
# for train_mode in "inp_layer" "out_layer" "adapter"
for train_mode in "adapter"
do
    name="gldv2_resnetgn_${train_mode}_${pfl_algo}_randinit_stateless_seed${seed}"
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
