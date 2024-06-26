python visualizer_anchored.py \
    --model "meta-llama/Llama-2-7b-chat-hf" \
    --anchors "./data/prompt-driven_benign.txt" "./data/prompt-driven_harmful.txt" \
    --datasets "./data/autodan_llama2_original.csv" "./data/autodan_llama2_jailbreak.csv" "./data/autodan_llama2_jailbreak_failed.csv" \
    --colors "bkgry" \
    --labels "Anchor Harmless" "Anchor Harmful" "Jailbreak Original" "Jailbreak Succeeded" "Jailbreak Failed" \
    --num_components 2 \
    --output ./visualization/pca_autodan_llama2-7b.png
