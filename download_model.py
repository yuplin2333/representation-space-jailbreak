"""
This script downloads the model and tokenizer from the Hugging Face Hub and saves
them to a local directory. This step is not necessary since they'll be downloaded
automatically when you run the experiment scripts.
"""

import os

import torch  # needed if you want to use torch.float16
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_CACHE"] = "/path/to/cache/"

tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm3-6b",
    cache_dir="/path/to/cache/",
)
model = AutoModelForCausalLM.from_pretrained(
    "THUDM/chatglm3-6b",
    cache_dir="/path/to/cache/",
    # device_map="auto",
    # low_cpu_mem_usage=True,
    # torch_dtype=torch.float16,
)

model.save_pretrained("/path/to/model/chatglm3-6b")
tokenizer.save_pretrained("/path/to/model/chatglm3-6b")

print("Done.")
