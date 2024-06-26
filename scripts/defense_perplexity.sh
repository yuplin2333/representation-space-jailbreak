#!/bin/bash

python defense_perplexity.py \
    --model_target "meta-llama/Llama-2-7b-chat-hf" \
    --result_source "./results/advbench_search_separateanchor_llama2-7b_direction_0_100.csv" \
    --model_judge "cais/HarmBench-Llama-2-13b-cls" \
    --output "./results/defense_perplexity_ours_llama2-7b.csv"
