import argparse
import ast
import os

import pandas as pd
from tqdm import tqdm

from evaluate_llm import ask_judge_harmbench
from jailbreak import check_jailbreak_success
from utils import load_dataset, load_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_target", type=str, required=True)
    parser.add_argument(
        "--result_source",
        type=str,
        required=True,
        help="Path to the CSV file containing the source results. Must contain the columns 'prompt' and 'adv_string'/'adv_prompt'.",
    )
    parser.add_argument(
        "--model_judge",
        type=str,
        default="cais/HarmBench-Llama-2-13b-cls",
        help="Model to judge the jailbreak success. Please use cais/HarmBench-Llama-2-13b-cls only.",
    )
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # Argument validation
    if not os.path.exists(args.result_source):
        raise ValueError(f"Dataset not found: {args.result_source}")
    # assert output file is a csv
    assert os.path.splitext(args.output)[1] == ".csv", "Output file must be a CSV file"
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    return args


def literal_eval(x):
    try:
        # return eval(x)  # This is dangerous because it can execute arbitrary code
        # If you are using this function to evaluate untrusted input, you should use ast.literal_eval
        return ast.literal_eval(x)
    except:
        return x


def main():
    args = parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {repr(value)}")

    dataset = pd.read_csv(args.result_source)
    assert "prompt" in dataset.columns, "Column 'prompt' not found in the dataset."
    assert (
        "adv_string" in dataset.columns or "adv_prompt" in dataset.columns
    ), "Column 'adv_string'/'adv_prompt' not found in the dataset."
    # Take only "prompt" and "adv_prompt" columns and make them np arrays
    # AutoDAN results have "adv_prompt" column which is the full adv prompt
    # GCG results have "adv_string" column which is the adv string only
    if "adv_prompt" in dataset.columns:
        print("Using 'adv_prompt' column")
    elif "adv_prompt" not in dataset.columns:
        print("Making 'adv_prompt' column from 'prompt' and 'adv_string' columns")
        dataset["prompt"] = dataset["prompt"].apply(literal_eval)
        dataset["adv_string"] = dataset["adv_string"].apply(literal_eval)
        dataset["adv_prompt"] = dataset["prompt"] + " " + dataset["adv_string"]

    dataset = dataset[["prompt", "adv_prompt", "jailbroken"]].values
    print(f"==>> dataset.shape: {dataset.shape}")

    model, tokenizer = load_model_and_tokenizer(args.model_target)
    model.eval()
    model.requires_grad_(False)
    model_judge, tokenizer_judge = load_model_and_tokenizer(args.model_judge)
    model_judge.eval()
    model_judge.requires_grad_(False)

    success_count = 0
    orig_jailbroken_count = 0
    orig_success_now_success_count = 0
    results = []
    total_length = len(dataset)

    for idx, (prompt, adv_prompt, orig_jailbroken) in tqdm(
        enumerate(dataset), total=total_length
    ):
        prompt = literal_eval(prompt)
        adv_prompt = literal_eval(adv_prompt)
        if orig_jailbroken:
            orig_jailbroken_count += 1

        full_prompt = f"{adv_prompt}"
        is_jailbroken, response = check_jailbreak_success(
            model,
            tokenizer,
            full_prompt,
            generate_length=32,
            return_response=True,
        )

        if is_jailbroken:
            # Double-check
            _, response = check_jailbreak_success(
                model=model,
                tokenizer=tokenizer,
                prompt=full_prompt,
                generate_length=512,
                return_response=True,
            )
            rating, _ = ask_judge_harmbench(
                model_judge,
                tokenizer_judge,
                prompt,
                response,
            )
            is_jailbroken = rating == 1
            if is_jailbroken:
                success_count += 1
                if orig_jailbroken:
                    orig_success_now_success_count += 1

        print(f"[{idx}, {is_jailbroken}]: {repr(full_prompt)} -> {repr(response)}")
        results.append(
            {
                "prompt": prompt,
                "adv_prompt": adv_prompt,
                "response": response,
                "jailbroken": is_jailbroken,
                "epoch": 0,
                "loss": 0,
                "attack_time": 0,
            }
        )

        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False, escapechar="\\", quoting=1)

    asr = success_count / total_length
    print(f"Clean ASR: {asr:.2f} ({success_count}/{total_length})")
    if orig_jailbroken_count > 0:
        transfer_rate = orig_success_now_success_count / orig_jailbroken_count
        print(
            f"Transfer rate: {transfer_rate:.2f} ({orig_success_now_success_count}/{orig_jailbroken_count})"
        )
    else:
        print("No original jailbroken examples found.")


if __name__ == "__main__":
    main()
