#!/bin/bash

#SBATCH --job-name=train
#SBATCH --comment="Train SO"
#SBATCH --partition=learnfair
#SBATCH --array=0-15  # TODO: count
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=32:00:00
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
logdir="/checkpoint/pillutla/pfl/outputs"
savedir="/checkpoint/pillutla/pfl/saved_models"

arch_params="\
            --num_attn_heads 2 \
            --num_transformer_layers 2 \
            --input_dim 128 \
            --attn_hidden_dim 64 \
            --fc_hidden_dim 512 \
            "
log_params="\
            --max_num_clients_for_logging 100 \
            --log_train_every_n_clients 1000 \
            --log_test_every_n_clients 5000"
train_params="\
            --train_batch_size 64  \
            --eval_batch_size 1024 \
            --central_optimizer adam \
            --use_warmup \
            --num_warmup_updates 5000 \
            --num_epochs_centralized 2 \
        "

common_args="${arch_params} ${log_params} ${train_params}"


# Populate the array
list_of_jobs=()
for do_tr in 0 0.1 0.2 0.3
do
for do_io in 0 0.1 0.2 0.3
do
name="so_tiny_dotr${do_tr}_doio${do_io}"
job=" --logfilename ${logdir}/${name} --savefilename ${savedir}/${name}.pt \
         --dropout_tr ${do_tr} --dropout_io ${do_io} "

list_of_jobs+=("${job}")
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

time python -u train_centralized.py \
            --dataset stackoverflow \
            ${common_args} \
            ${job}

echo "Job completed at $(date)"
