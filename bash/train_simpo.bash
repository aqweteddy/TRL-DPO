
export LAUNCHER="accelerate launch \
    --config_file cfg/zero2.yml \
    --num_processes 4 \
    --num_machines 1 \
    --main_process_port 29500 \
    --rdzv_backend c10d \
    "

export DATASET="{'path':'json','data_files':'/volume/finance-data/processed-data/rag_ft/dpo_v0/total.jsonl'}"
export MODEL=/volume/models/Qwen/Qwen2.5-1.5B-Instruct/
export OUTPUT_MODEL_PATH="/volume/models/test/qwen-test/"

export PYTHON_FILE="./script/train_simpo.py"

export SCRIPT_ARGS=" \
    --dataset_name $DATASET \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT_MODEL_PATH \
    --beta 0.4 \
    --bf16 True \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
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
    --max_length 16384 \
    --max_prompt_length 8192 \
    --max_completion_length 4096 \
    --truncation_mode keep_end \
    --num_train_epochs 2 \
    --report_to wandb \
    --gradient_checkpointing True \
    --dataset_num_proc 8 \
    --remove_unused_columns False"
    # --use_peft \
    # --lora_r=16 \
    # --lora_alpha=32"


# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $SCRIPT_ARGS" 
echo $CMD
$CMD