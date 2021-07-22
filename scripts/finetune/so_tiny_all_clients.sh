#!/bin/bash

#SBATCH --job-name=train
#SBATCH --comment="Finetune SO"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=0-12  # TODO: count
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=13:00:00
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600

source ~/.bashrc  # load all modules
source activate pyt19  # load environment

echo "Running [ ${0} ${@} ] on $(hostname), starting at $(date)"
echo "Job id = ${SLURM_JOB_ID}, task id = ${SLURM_ARRAY_TASK_ID}"
echo "PWD = $(pwd)"
echo "python path = $(which python)"

set -exu

model_size="tiny"

# Default arguments
logdir="/checkpoint/pillutla/pfl/outputs"
modelfilename="/checkpoint/pillutla/pfl/saved_models/so_${model_size}_try1.pt"

arch_params="--arch_size tiny"
train_params="\
            --train_batch_size 32 \
            --eval_batch_size 1024 \
            --optimizer adam \
            --scheduler linear \
            --lr 3.5e-4 \
        "
common_params="\
            --num_updates_personalization 100 \
            --modelfilename ${modelfilename} \
            --max_num_clients_for_personalization 1000 \
        "

list_of_jobs=()
for train_mode in "finetune" "finetune_inp_layer" "finetune_out_layer"
do
    name="so_${model_size}_try1_${train_mode}"
    task_params="--train_mode ${train_mode} --logfilename ${logdir}/${name}"
    list_of_jobs+=("${task_params}")
done

# finetune 
train_mode="finetune_tr_layer"
for layers in "0" "1" "0 1" 
do
    l2=`echo ${layers} | sed 's/ /+/g'`
    name="so_${model_size}_try1_${train_mode}_${l2}"
    task_params="--train_mode ${train_mode} --layers_to_finetune ${layers} --logfilename ${logdir}/${name}"
    list_of_jobs+=("${task_params}")
done

# adapter
train_mode="adapter"
for hidden_dim in 2 4 8 16 32 64 128
do
    name="so_${model_size}_try1_${train_mode}_${hidden_dim}"
    task_params="--train_mode ${train_mode} --adapter_hidden_dim ${hidden_dim} --logfilename ${logdir}/${name}"
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

time python -u train_personalized_finetune.py \
            --dataset stackoverflow \
            ${train_params}  \
            ${common_params}  \
            ${arch_params}  \
            ${args} 

echo "Job completed at $(date)"