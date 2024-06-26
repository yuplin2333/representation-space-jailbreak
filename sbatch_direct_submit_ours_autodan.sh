#!/bin/bash

base_output_dir=./results
base_log_dir=./log
model_repos=("meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf" "meta-llama/Meta-Llama-3-8B-Instruct" "lmsys/vicuna-7b-v1.5" "google/gemma-7b-it")
model_nicknames=("llama2-7b" "llama2-13b" "llama3-8b" "vicuna" "gemma")
autodan_model_short_names=("Llama-2" "Llama-2" "Llama-3" "Vicuna" "Gemma")
autodan_model_developers=("Meta" "Meta" "Meta" "LMSYS" "Google")
start=0
end=520
step=20

echo "Submitting ours_autodan"
for i in "${!model_repos[@]}"; do
    model_repo=${model_repos[$i]}
    model_nickname=${model_nicknames[$i]}
    autodan_model_short_name=${autodan_model_short_names[$i]}
    autodan_model_developer=${autodan_model_developers[$i]}
    echo "Submitting ${model_nickname}, ${autodan_model_short_name} by ${autodan_model_developer}"
    for start_idx in $(seq "$start" "$step" "$((end - 1))"); do
        end_idx=$(($start_idx + $step))
        if [ "$end_idx" -gt "$end" ]; then
            end_idx="$end"
        fi
        output_dir="$base_output_dir/ours_autodan/${model_nickname}"
        job_name="ours_autodan_${model_nickname}_${start_idx}_${end_idx}"
        log_path="$base_log_dir/attack/ours_autodan/${model_nickname}/idx_${start_idx}_${end_idx}.log"
        ####### MODIFY THE FOLLOWING LINE TO YOUR NEEDS #######
        sbatch --job-name=$job_name --ntasks=1 --cpus-per-task=8 --gpus=a100:1 --mem=32G --time=0-04:00:00 --output=$log_path \
            scripts/directrun_ours_autodan.sh $model_repo $autodan_model_short_name $autodan_model_developer $start_idx $end_idx $output_dir
    done
done
