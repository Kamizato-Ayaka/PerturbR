#!/bin/bash
set -euo pipefail


LOCAL_DATA_DIR="/home/aiscuser/dataset"
mkdir -p "${LOCAL_DATA_DIR}"

############################
# 实验列表配置
############################
MODELS=(
    "Qwen/Qwen3-1.7B-base"
)

DATASETS=(
  "/root/dengjie/AI4SCI/PP-data/GSE92742-SFT/Reranker-SFT.jsonl"
)

OUTPUTS=(
    "/root/dengjie/AI4SCI/Model-Saves/Qwen3-1.7B-SFT-Reranker-Full"
)

############################
# 串行执行
############################
NUM_EXP=${#DATASETS[@]}

for ((i=0; i<NUM_EXP; i++)); do

  echo "======================================"
  echo "Starting Experiment $i"
  echo "Model:   ${MODELS[$i]}"
  echo "Dataset: ${DATASETS[$i]}"
  echo "Output:  ${OUTPUTS[$i]}"
  echo "======================================"

  mkdir -p "${OUTPUTS[$i]}"

  ############################
  # 1️⃣ rsync dataset 到 /home/aiscuser/dataset
  ############################
  SRC_DATASET="${DATASETS[$i]}"


  ############################
  # 2️⃣ 使用本地路径训练
  ############################
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NPROC_PER_NODE=4 \
  swift sft \
      --model "${MODELS[$i]}" \
      --tuner_type lora \
      --dataset "${SRC_DATASET}" \
      --torch_dtype float16 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --learning_rate 2e-5 \
      --gradient_accumulation_steps 16 \
      --save_total_limit 3 \
      --deepspeed zero3 \
      --logging_steps 5 \
      --save_only_model true \
      --output_dir "${OUTPUTS[$i]}" \
      --warmup_ratio 0.05 \
      --dataset_num_proc 64

  echo "Experiment $i finished"
  echo ""

done

echo "All experiments completed."