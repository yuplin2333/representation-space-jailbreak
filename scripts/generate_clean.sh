#!/bin/bash

models=("llama2-7b" "llama2-13b" "llama3-8b" "vicuna" "gemma")
model_params=("meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf" "meta-llama/Meta-Llama-3-8B-Instruct" "lmsys/vicuna-7b-v1.5" "google/gemma-7b-it")

# Loop over each model
for i in "${!models[@]}"; do
    model="${models[i]}"
    model_param="${model_params[i]}"

    python generate_clean_response.py \
        --model "$model_param" \
        --dataset "./data/advbench/harmful_behaviors.csv" \
        --column "goal" \
        --idx 0 520 \
        --model_judge "cais/HarmBench-Llama-2-13b-cls" \
        --output "./results/clean/${model}.csv" 2>&1 | tee "./log/clean/${model}.log"
done
