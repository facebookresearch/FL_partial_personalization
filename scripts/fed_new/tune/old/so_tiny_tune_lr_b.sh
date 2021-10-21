#!/bin/bash

#SBATCH --job-name=train
#SBATCH --comment="SO lr"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=0-11  # TODO: count
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
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
logdir="/checkpoint/pillutla/pfl/outputs2"
savedir="/checkpoint/pillutla/pfl/saved_models2"
common_args=" --dataset stackoverflow --train_batch_size 64 --eval_batch_size 1024 "
common_args="${common_args} --num_communication_rounds 500 --num_clients_per_round 10 --num_local_epochs 1"
common_args="${common_args} --clip_grad_norm  --log_test_every_n_rounds 100 --max_num_clients_for_logging 1000 "  # TODO!
common_args="${common_args} --arch_size ${arch_size} \
        --server_optimizer adam --client_scheduler const --client_lr 10 \
        --client_optimizer sgd --global_scheduler linear --global_warmup_fraction 0.1 \
        --pretrained_model_path /checkpoint/pillutla/pfl/saved_models2/so_tiny_pretrain_1000/checkpoint.pt \
        "

# Populate the array
list_of_jobs=()


for server_lr in 1e-3 1e-4
do
for pfl_algo in "pfl_joint" "pfl_alternating"
do

train_mode="tr_layer"
for layers in "0" "1"  
do
    l2=`echo ${layers} | sed 's/ /+/g'`
    name="so_${arch_size}_${train_mode}_${l2}_${pfl_algo}_pretrained_lr10_slr${server_lr}"
    task_params="--server_lr ${server_lr} --pfl_algo ${pfl_algo} --personalize_on_client ${train_mode} --layers_to_finetune ${layers} --logfilename ${logdir}/${name} --savedir ${savedir}/${name}"
    list_of_jobs+=("${task_params}")
done

train_mode="adapter"
for hidden_dim in 16
do
    name="so_${arch_size}_${train_mode}_${hidden_dim}_${pfl_algo}_pretrained_lr10_slr${server_lr}"
    task_params="--server_lr ${server_lr} --pfl_algo ${pfl_algo} --personalize_on_client ${train_mode} --adapter_hidden_dim ${hidden_dim} --logfilename ${logdir}/${name} --savedir ${savedir}/${name}"
    list_of_jobs+=("${task_params}")
done

done  # pfl algo
done  # lr

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