"""
This script is deprecated. We keep it here for reference only.
"""


import argparse
import ast
import os

import pandas as pd

from jailbreak import check_jailbreak_success
from utils import load_model_and_tokenizer, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the result of a jailbreak")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--result_csv",
        type=str,
        required=True,
        help="Path to the attack result csv. "
        "User prompt should be in the column 'prompt', and "
        "adv string should be in the column 'adv_string'.",
    )
    parser.add_argument(
        "--generate_length",
        type=int,
        default=512,
        help="Max length of the model generating the response. "
        "Larger length may lead to better results but slower speed.",
    )
    parser.add_argument(
        "--prefix_mode",
        action="store_true",
        help="If set, use prefix mode to generate the response instead of suffix.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output csv"
    )
    args = parser.parse_args()

    if not os.path.exists(args.result_csv):
        raise FileNotFoundError(f"result_csv {args.result_csv} not found")
    assert os.path.splitext(args.output)[1] == ".csv", "Output file must be a CSV file"
    assert not os.path.exists(
        args.output
    ), f"Output file {args.output} already exists. Please remove it first."

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    return args


def literal_eval(x):
    try:
        # return eval(x)  # This is dangerous because it can execute arbitrary code
        # If you are using this function to evaluate untrusted input, you should use ast.literal_eval
        return ast.literal_eval(x)
    except Exception:
        return x


def eval_df_result_success(model, tokenizer, result_df, generate_length, prefix_mode):
    total_length = len(result_df)
    success_count = 0
    results = []

    for i, row in result_df.iterrows():
        prompt = literal_eval(row["prompt"])
        adv_string = literal_eval(row["adv_string"])
        prompt_with_adv_string = (
            f"{prompt} {adv_string}" if not prefix_mode else f"{adv_string} {prompt}"
        )
        jailbroken, response = check_jailbreak_success(
            model,
            tokenizer,
            prompt_with_adv_string,
            generate_length=generate_length,
            return_response=True,
        )
        if jailbroken:
            success_count += 1

        results.append(
            {
                "prompt": repr(prompt),
                "adv_string": repr(adv_string),
                "prompt_with_adv_string": repr(prompt_with_adv_string),
                "response": repr(response),
                "jailbroken": jailbroken,
            }
        )

        print(f"{i}/{total_length}".center(50, "-"))
        print(f"==>> prompt: {repr(prompt)}")
        print(f"==>> adv_string: {repr(adv_string)}")
        print(f"==>> prompt_with_adv_string: {repr(prompt_with_adv_string)}")
        print(f"==>> response: {repr(response)}")
        print(f"==>> jailbroken: {jailbroken}")
        print()

    asr = success_count / total_length
    print(f"Success Rate: {asr:.4f} ({success_count}/{total_length})")

    return results, asr


def main():
    args = parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {repr(value)}")
    set_seed(args.seed)

    result_df = pd.read_csv(args.result_csv)
    assert "prompt" in result_df.columns, "'prompt' column not found in the result csv"
    assert (
        "adv_string" in result_df.columns
    ), "'adv_string' column not found in the result csv"

    model, tokenizer = load_model_and_tokenizer(args.model)
    results, _ = eval_df_result_success(
        model, tokenizer, result_df, args.generate_length, args.prefix_mode
    )

    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
