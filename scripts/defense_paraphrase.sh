#!/bin/bash

python defense_paraphrase.py \
    --model_target "meta-llama/Llama-2-7b-chat-hf" \
    --result_source "./results/advbench_gcg_harmbench_0_100.csv" \
    --model_judge "cais/HarmBench-Llama-2-13b-cls" \
    --output "./results/defense_paraphrase_gcg_llama2-7b.csv"
