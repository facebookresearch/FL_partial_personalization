#!/bin/bash

#SBATCH --job-name=train
#SBATCH --comment="SO mini lr"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=0-39%14  # TODO: count
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=3:30:00
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
logdir="/checkpoint/pillutla/pfl/outputs3"
savedir="/checkpoint/pillutla/pfl/saved_models3"
common_args=" --dataset stackoverflow --train_batch_size 64 --eval_batch_size 1024 "
common_args="${common_args} --clip_grad_norm  --arch_size ${arch_size} --lr 0.1 \
        --pretrained_model_path /checkpoint/pillutla/pfl/saved_models2/so_${arch_size}_pretrain_1000/checkpoint.pt \
    "

# Populate the array
list_of_jobs=()

for seed in 2 3 4 5 
do
for ne in 1 5 
do
train_mode="finetune"
    name="so_${arch_size}_fedavg_${train_mode}_ne${ne}_seed${seed}"
    task_params=" --num_epochs_personalization  ${ne} \
        --seed ${seed} --personalize_on_client ${train_mode} \
        --logfilename ${logdir}/${name} --savedir ${savedir}/${name}"
    list_of_jobs+=("${task_params}")

train_mode="tr_layer"
for layers in "0" "3"  
do
    l2=`echo ${layers} | sed 's/ /+/g'`
    name="so_${arch_size}_fedavg_${train_mode}_${l2}_ne${ne}_seed${seed}"
    task_params=" --num_epochs_personalization  ${ne} \
        --seed ${seed} --personalize_on_client ${train_mode} --layers_to_finetune ${layers}  \
        --logfilename ${logdir}/${name} --savedir ${savedir}/${name}"
    list_of_jobs+=("${task_params}")
done

train_mode="adapter"
for hidden_dim in 16 64
do
    name="so_${arch_size}_fedavg_${train_mode}_${hidden_dim}_ne${ne}_seed${seed}"
    task_params=" --num_epochs_personalization  ${ne} \
        --seed ${seed} --personalize_on_client ${train_mode} --adapter_hidden_dim ${hidden_dim}  \
        --logfilename ${logdir}/${name} --savedir ${savedir}/${name}"
    list_of_jobs+=("${task_params}")
done
done  # ne
done # seed


# Run
num_jobs=${#list_of_jobs[@]}
job_id=${SLURM_ARRAY_TASK_ID}
if [ ${job_id} -ge ${num_jobs} ] ; then
    echo "Invalid job id; qutting"
    exit 2
fi
echo "-------- STARTING JOB ${job_id}/${num_jobs}"
args=${list_of_jobs[${job_id}]}

# sleep 10m

sleep ${job_id}m

time python -u train_finetune.py \
        ${common_args} \
        ${args}

echo "Job completed at $(date)"
