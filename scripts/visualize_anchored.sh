python visualizer_anchored.py \
    --model "meta-llama/Llama-2-7b-chat-hf" \
    --anchors "./data/prompt-driven_benign.txt" "./data/prompt-driven_harmful.txt" \
    --datasets "./data/visualization/gcg_llama2-7b_original.csv" "./data/visualization/gcg_llama2-7b_jailbreak.csv" "./data/avisualization/gcg_llama2-7b_jailbreak_failed.csv" \
    --text_columns "source" "source" "source" \
    --colors "bkgry" \
    --labels "Anchor Harmless" "Anchor Harmful" "Jailbreak Original" "Jailbreak Succeeded" "Jailbreak Failed" \
    --num_components 2 \
    --output ./visualization/pca_gcg_llama2-7b.png
