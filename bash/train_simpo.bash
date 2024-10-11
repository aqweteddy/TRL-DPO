#!/bin/bash

export LAUNCHER="accelerate launch \
    --config_file cfg/zero2.yml \
    --num_processes 4 \
    --num_machines 1 \
    --main_process_port 29500 \
    --rdzv_backend c10d \
    "

export DATASET="{'path':'aqweteddy/mrc','revision':'spin-v0'}"
export MODEL=/workspace/TRL-DPO/llama3.2-3b-instruct_ft-v2c-e2
export OUTPUT_MODEL_PATH="/workspace/TRL-DPO/ckpt/llama3.2-3b-instruct_ft-v2c-e2_dpo-v0"

export PYTHON_FILE="./script/train_simpo.py"

export SCRIPT_ARGS=" \
    --dataset_name $DATASET \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT_MODEL_PATH \
    --beta 2.5 \
    --bf16 True \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.01 \
    --loss_type simpo \
    --cpo_alpha 0.5 \
    --simpo_gamma 0.5 \
    --attn_implementation flash_attention_2 \
    --optim adamw_8bit \
    --save_only_model \
    --dataloader_pin_memory False \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --save_strategy epoch \
    --save_safetensors \
    --max_length 16384 \
    --max_prompt_length 12000 \
    --max_completion_length 4096 \
    --truncation_mode keep_end \
    --num_train_epochs 2 \
    --logging_steps 10 \
    --report_to wandb \
    --gradient_checkpointing True \
    --dataset_num_proc 8 \
    --remove_unused_columns False"



# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $SCRIPT_ARGS" 
echo $CMD
$CMD