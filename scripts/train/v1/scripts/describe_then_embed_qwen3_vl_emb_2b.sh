export INFONCE_TEMPERATURE=0.05
# 参与前 K 个负例（按相似度从高到低，前 K 个即最难 K 个）；16 = 全部
export INFONCE_HARD_NEGATIVES=16
export INFONCE_USE_BATCH=false
# λ：生成式损失的相对学习率权重（L = λ·L_gen + L_contra，生成式学习需要更大学习率）
export DESC_EMBED_LAMBDA=5
# <gen_emb> 行注意力模式：default / caption_only / image_only
export GEN_EMB_ATTN=caption_only
# Doc 侧生成监督范围：all（正+负例描述都监督）/ positive（仅正例）/ none
export DOC_GEN_SUPERVISE=positive
# L_gen 参与训练的步数比例（1.0=全程；0.2=仅前 20% 步，之后纯对比）
export GEN_LOSS_RATIO=0.5
# 序列末尾连续 <gen_emb> token 数（>=1）
export GEN_EMB_NUM=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== 路径（按需修改；默认相对本脚本定位，可在任意目录执行）=====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=${MODEL:-/home/qkh/models/Qwen--Qwen3-VL-Embedding-2B}
PLUGIN=${PLUGIN:-${SCRIPT_DIR}/describe_then_embed_plugin.py}
DATA=${DATA:-/home/qkh/FunEmbed/20261027_model_train/v1/data/describe_then_embed_train.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/home/qkh/FunEmbed/20261027_model_train/v1/models/caption_only}

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model ${MODEL} \
    --model_type qwen3_vl_emb \
    --task_type embedding \
    --tuner_type lora \
    --loss_type describe_then_embed \
    --template describe_then_embed \
    --new_special_tokens '<gen_emb>' \
    --dataset ${DATA} \
    --custom_register_path ${PLUGIN} \
    --padding_free false \
    --torch_dtype bfloat16 \
    --max_length 8192 \
    --gradient_checkpointing true \
    --split_dataset_ratio 0.02 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --num_train_epochs 3 \
    --save_strategy epoch \
    --save_steps 200 \
    --eval_strategy steps \
    --eval_steps 0.1 \
    --eval_metric describe_then_embed \
    --logging_steps 20 \
    --save_total_limit 5 \
    --target_modules all-linear \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --output_dir ${OUTPUT_DIR} \
    --deepspeed zero2
