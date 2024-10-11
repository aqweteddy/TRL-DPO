
export LAUNCHER="accelerate launch \
    --config_file cfg/zero3.yml \
    --num_processes 8 \
    --num_machines 1 \
    --main_process_port 29500 \
    --rdzv_backend c10d \
    "
# --rdzv_backend c10d \
export DATASET="{'path':'json','data_files':'/root/SPIN/result/*.jsonl'}"
export MODEL=/volume/models/taishin_ckpt/yi-1.5-9b-chat_cp-v1.8-full/iter_15000.hf
export OUTPUT_MODEL_PATH="/volume/models/taishin_ckpt/yi-1.5-9b-chat_cp-v1.8-full_spin-ft0"

export PYTHON_FILE="./script/train_dpo.py"

export SCRIPT_ARGS=" \
    --beta 0.4 \
    --bf16 True \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.1 \
    --loss_type sppo_hard \
    --attn_implementation flash_attention_2 \
    --optim adamw_8bit \
    --model_name_or_path $MODEL \
    --save_only_model \
    --dataset_name $DATASET \
    --dataloader_pin_memory False \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --save_strategy epoch \
    --max_prompt_length 3000 \
    --max_length 4096 \
    --num_train_epochs 2 \
    --report_to wandb \
    --gradient_checkpointing True \
    --dataset_num_proc 8 \
    --remove_unused_columns False \
    --output_dir $OUTPUT_MODEL_PATH \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=32"


# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $SCRIPT_ARGS" 
$CMD
#echo $CMD