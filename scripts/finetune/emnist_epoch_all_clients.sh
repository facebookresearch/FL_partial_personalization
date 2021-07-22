#!/bin/bash

#SBATCH --job-name=train
#SBATCH --comment="Finetune EMNIST"
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/pillutla/pfl/outs/%A_%a.out
#SBATCH --array=0-5  # TODO: count
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

num_epochs=3

# Default arguments
logdir="/checkpoint/pillutla/pfl/outputs"
modelfilename="/checkpoint/pillutla/pfl/saved_models/emnist_resnet.pt"

# lr = final learning rate at the end of centralized training 
train_params="\
            --train_batch_size 32 \
            --eval_batch_size 256 \
            --optimizer sgd \
            --scheduler const_and_cut \
            --lr 5e-4 \
            --lr_decay_factor 0.5 \
            --lr_decay_every 25
        "
common_params="\
            --num_epochs_personalization ${num_epochs} \
            --use_epochs_for_personalization \
            --modelfilename ${modelfilename} \
            --max_num_clients_for_personalization 1114 \
        "

list_of_jobs=()
for train_mode in "finetune" "finetune_inp_layer" "finetune_out_layer" "adapter"
do
    name="emnist_e${num_epochs}_${train_mode}"
    task_params="--train_mode ${train_mode} --logfilename ${logdir}/${name}"
    list_of_jobs+=("${task_params}")
done

# finetune 
train_mode="finetune_res_layer"  # layers are from [1, 2, 3, 4]
for layers in "1 2" "3 4"
do
    l2=`echo ${layers} | sed 's/ /+/g'`
    name="emnis_e${num_epochs}_${train_mode}_${l2}"
    task_params="--train_mode ${train_mode} --layers_to_finetune ${layers} --logfilename ${logdir}/${name}"
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
            --dataset emnist \
            --model_name resnet \
            ${train_params}  \
            ${common_params}  \
            ${args} 

echo "Job completed at $(date)"