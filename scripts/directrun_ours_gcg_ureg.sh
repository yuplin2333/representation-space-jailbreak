#!/bin/bash

model_repo=$1
start_idx=$2
end_idx=$3
output_dir=$4

python jailbreak_ours_gcg.py \
    --model "${model_repo}" \
    --dataset "data/advbench/harmful_behaviors.csv" \
    --column "goal" \
    --idx ${start_idx} ${end_idx} \
    --anchor_datasets "data/prompt-driven_benign.txt" "./data/prompt-driven_harmful.txt" \
    --ureg_norm_coef 0 \
    --init_adv_string "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !" \
    --max_epochs 500 \
    --sampling_number 512 \
    --model_judge "cais/HarmBench-Llama-2-13b-cls" \
    --output_dir "${output_dir}/" \
    --output_plot_dir "${output_dir}/plot/"
