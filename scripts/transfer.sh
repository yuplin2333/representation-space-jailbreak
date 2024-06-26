#!/bin/bash

python generate_transfer_response.py \
    --model_target "google/gemma-7b-it" \
    --result_source "./results/advbench_search_harmbench_0_100.csv" \
    --model_judge "cais/HarmBench-Llama-2-13b-cls" \
    --output "./results/transfer_search_llama2-7b_gemma.csv"
