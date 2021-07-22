#!/bin/bash
set -exu

# Default arguments
logdir="/checkpoint/pillutla/pfl/outputs"
modelfilename="/checkpoint/pillutla/pfl/saved_models/so_tiny_try1.pt"

arch_params="\
            --num_attn_heads 2 \
            --num_transformer_layers 2 \
            --input_dim 128 \
            --attn_hidden_dim 64 \
            --fc_hidden_dim 512 \
            --dropout_tr 0 \
            --dropout_io 0 \
            "
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
        "

list_of_jobs=()
for train_mode in "finetune" "finetune_inp_layer" "finetune_out_layer"
do
    name="so_tiny_try1_${train_mode}"
    task_params="--train_mode ${train_mode} --logfilename ${logdir}/${name}"
    time python -u train_personalized_finetune.py \
                --dataset stackoverflow \
                ${train_params}  \
                ${common_params}  \
                ${arch_params}  \
                ${task_params}  > logs/${name} 2>&1
done

train_mode="finetune_tr_layer"
for layers in "0" "1" "0 1" 
do
    l2=`echo ${layers} | sed 's/ /+/g'`
    name="so_tiny_try1_${train_mode}_${l2}"
    task_params="--train_mode ${train_mode} --layers_to_finetune ${layers} --logfilename ${logdir}/${name}"
    time python -u train_personalized_finetune.py \
                --dataset stackoverflow \
                ${train_params}  \
                ${common_params}  \
                ${arch_params}  \
                ${task_params}  > logs/${name} 2>&1
done


echo "Job completed at $(date)"