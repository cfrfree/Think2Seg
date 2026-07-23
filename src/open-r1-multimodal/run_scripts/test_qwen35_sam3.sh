source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3

export MODEL_NAME=/root/nas/model_cache/modelscope/models/Qwen/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b
export EARTHREASON_ROOT=/root/nas/think2seg/EarthReason/datasets--earth-insights--EarthReason/snapshots/983ff4339fa28e8ba2f87700fb783e5a4a2c462d
export SAM_VERSION=sam3
export SAM_ROOT=/root/sam3
export RUN_NAME=TEST-Qwen3.5-4B-SAM3-$(date +%Y%m%d_%H%M%S)
export CUDA_VISIBLE_DEVICES=0,1

export SWANLAB_API_KEY=XrpnxvyT4wxVFtTWMXS5Z
export SWANLAB_PROJECT=Think2Seg-RS

export DEBUG_MODE="true"
export LOG_PATH="/root/nas/think2seg/output_ultra/$RUN_NAME/debug_log.txt"

export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=32

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_geo_ultra.py \
    --deepspeed local_scripts/zero2.json \
    --dataset_name none \
    --output_dir /root/nas/think2seg/output_ultra/$RUN_NAME \
    --model_name_or_path $MODEL_NAME \
    --max_prompt_length 512 \
    --max_completion_length 64 \
    --num_generations 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to none \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --max_steps 3 \
    --run_name $RUN_NAME \
    --sam_version $SAM_VERSION \
    --sam_root $SAM_ROOT \
    --sam_device cuda:1 \
    --earthreason_root $EARTHREASON_ROOT \
    --save_steps 100 \
    --save_only_model true \
    --save_total_limit 2 \
    --use_datasets earthreason \
    --freeze_vision_modules true \
    --use_peft \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_modules "q_proj,v_proj" \
    --beta 0.0 \
