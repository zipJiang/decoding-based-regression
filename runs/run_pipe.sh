#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=a100

ENV_PATH="/scratch/bvandur1/zjiang31/decoding-based-regression/.env"
NUM_LABELS=10
MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
REGULARIZATION="null"
SCALE_FACTOR=1.0

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --num-labels)
    # MODIFIY TO ALLOW MULTIPLE CONFIG PATHS
    # CONFIG_PATH="$2"
    NUM_LABELS="$2"
    shift
    shift
    ;;
    --model-name)
    MODEL_NAME="$2"
    shift
    shift
    ;;
    --reg-method)
    REGULARIZATION="$2"
    shift
    shift
    ;;
    --scale)
    SCALE_FACTOR="$2"
    shift
    shift
    ;;
esac
done

export NUM_LABELS
export MODEL_NAME
export REGULARIZATION
export SCALE_FACTOR

# first run the data generation script

# for task_name in "unli" "ecare" "entailmentbank" "gnli"
# do
#     export TASK_NAME=$task_name
#     /scratch/bvandur1/zjiang31/decoding-based-regression/runs/run_task.sh \
#         --config-path /scratch/bvandur1/zjiang31/decoding-based-regression/configs/dataset/dataset_preparation.jsonnet
# done

# # then run the resizing script
# /scratch/bvandur1/zjiang31/decoding-based-regression/runs/run_task.sh \
#     --config-path /scratch/bvandur1/zjiang31/decoding-based-regression/configs/preprocess/resize_embeddings.jsonnet

# then run the training script
/weka/scratch/bvandur1/zjiang31/decoding-based-regression/runs/run_task.sh \
    --config-path /weka/scratch/bvandur1/zjiang31/decoding-based-regression/configs/training/sft_regression.jsonnet \
    --use-accelerate

# then run the evaluation script
/weka/scratch/bvandur1/zjiang31/decoding-based-regression/runs/run_task.sh \
    --config-path /weka/scratch/bvandur1/zjiang31/decoding-based-regression/configs/eval/evaluation.jsonnet