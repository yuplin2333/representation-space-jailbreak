#!/bin/bash

model_repo=$1
model_short_name=$2
developer_name=$3
start_idx=$4
end_idx=$5
output_dir=$6

python jailbreak_autodan.py \
    --model "${model_repo}" \
    --model_short_name ${model_short_name} \
    --developer_name ${developer_name} \
    --dataset "data/advbench/harmful_behaviors.csv" \
    --column "goal" \
    --idx ${start_idx} ${end_idx} \
    --anchor_datasets "data/prompt-driven_benign.txt" "./data/prompt-driven_harmful.txt" \
    --model_mutate "mistralai/Mistral-7B-Instruct-v0.2" \
    --max_epochs 100 \
    --batch_size 64 \
    --model_judge "cais/HarmBench-Llama-2-13b-cls" \
    --output_dir "${output_dir}/" \
    --output_plot_dir "${output_dir}/plot/"
