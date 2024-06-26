#!/bin/bash

# model_target: "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", ...
# Refer to https://platform.openai.com/docs/models/

python jailbreak_dan_api.py \
    --model "gpt-4-turbo-preview" \
    --dataset "./data/advbench/harmful_behaviors.csv" \
    --column "goal" \
    --idx 0 520 \
    --model_judge "cais/HarmBench-Llama-2-13b-cls" \
    --output "./results/advbench_dan_gpt-4_${1}_${2}.csv"
