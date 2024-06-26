#!/bin/bash

# model_target: "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", ...
# Refer to https://platform.openai.com/docs/models/

python generate_transfer_response_api.py \
    --model_target "gpt-3.5-turbo" \
    --result_source "./results/advbench_gcg_harmbench_0_100.csv" \
    --model_judge "cais/HarmBench-Llama-2-13b-cls" \
    --output "./results/transfer_gcg_llama2-7b_gpt-3.5.csv"
