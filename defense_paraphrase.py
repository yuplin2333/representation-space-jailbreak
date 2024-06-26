import argparse
import ast
import os

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from evaluate_llm import ask_judge_harmbench
from jailbreak import check_jailbreak_success
from utils import load_model_and_tokenizer


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


# def gpt_paraphrase(original_text, prompt="paraphrase the following sentences:", paraphrase_model_name=None, paraphrase_temperature=None, paraphrase_max_tokens=None):
#     # https://github.com/neelsjain/baseline-defenses/blob/main/paraphrase.py
#     assert prompt, "Prompt must be provided for GPT attack"

#     paraphrase_query = prompt + original_text
#     query_msg = {"role": "user", "content": paraphrase_query}

#     from tenacity import retry, stop_after_attempt, wait_random_exponential

#     @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(25))
#     def completion_with_backoff(model, messages, temperature, max_tokens):
#         return openai.ChatCompletion.create(
#             model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
#         )

#     outputs = completion_with_backoff(
#         model=paraphrase_model_name,
#         messages=[query_msg],
#         temperature=paraphrase_temperature,
#         max_tokens=paraphrase_max_tokens,
#     )

#     paraphrase_prompt = outputs.choices[0].message.content

#     return paraphrase_prompt


def gpt_paraphrase(
    original_text,
    prompt,
    paraphrase_model_name,
    paraphrase_temperature,
    paraphrase_max_tokens,
):
    assert prompt, "Prompt must be provided for GPT attack."

    messages = [{"role": "user", "content": prompt + original_text}]

    # If you're submitting a slurm job, the API key from OS env won't be available.
    # In that case, manually read ./.env file and remove the "OPENAI_API_KEY=" prefix.
    try:
        client = OpenAI()
    except Exception as e:
        with open(".env", "r") as f:
            api_key = f.readline().strip().replace("OPENAI_API_KEY=", "")
        try:
            client = OpenAI(api_key=api_key)
        except Exception as e:
            print("Error: ", e)
            print(
                "Cannot create OpenAI client. Your API key has issue. \n"
                "If you haven't set an API key, please refer to ./.env_template "
                "and create a ./.env file to put your OpenAI API key in."
            )
            exit(1)

    completion = client.chat.completions.create(
        model=paraphrase_model_name,
        messages=messages,
        temperature=paraphrase_temperature,
        max_tokens=paraphrase_max_tokens,
    )
    response = completion.choices[0].message.content

    return response


def main():
    args = parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {repr(value)}")

    dataset = pd.read_csv(args.result_source)
    assert "prompt" in dataset.columns, "Column 'prompt' not found in the dataset."
    assert (
        "adv_string" in dataset.columns or "adv_prompt" in dataset.columns
    ), "Column 'adv_string'/'adv_prompt' not found in the dataset."
    # Take only "prompt" and "adv_string" columns and make them np arrays
    # AutoDAN results have "adv_prompt" column which is the full adv prompt
    # GCG results have "adv_string" column which is the adv string only
    if "adv_prompt" in dataset.columns:
        print("Using 'adv_prompt' column")
    elif "adv_prompt" not in dataset.columns:
        print("Making 'adv_prompt' column from 'prompt' and 'adv_string' columns")
        dataset["prompt"] = dataset["prompt"].apply(literal_eval)
        dataset["adv_string"] = dataset["adv_string"].apply(literal_eval)
        dataset["adv_prompt"] = dataset["prompt"] + " " + dataset["adv_string"]
    dataset = dataset[["prompt", "adv_prompt"]].values
    print(f"==>> dataset.shape: {dataset.shape}")

    model, tokenizer = load_model_and_tokenizer(args.model_target)
    model.eval()
    model.requires_grad_(False)
    model_judge, tokenizer_judge = load_model_and_tokenizer(args.model_judge)
    model_judge.eval()
    model_judge.requires_grad_(False)

    success_count = 0
    results = []
    total_length = len(dataset)

    for idx, (prompt, adv_prompt) in tqdm(enumerate(dataset), total=total_length):
        prompt = literal_eval(prompt)
        adv_string = literal_eval(adv_prompt)
        full_prompt = f"{adv_string}"

        # Defense: Paraphrase the prompt
        # Parameters following Baseline-Defense paper
        paraphrased_prompt = gpt_paraphrase(
            original_text=full_prompt,
            prompt="paraphrase the following sentences:",
            paraphrase_model_name="gpt-3.5-turbo",
            paraphrase_temperature=0.7,
            paraphrase_max_tokens=100,
        )

        is_jailbroken, response = check_jailbreak_success(
            model,
            tokenizer,
            paraphrased_prompt,
            generate_length=32,
            return_response=True,
        )

        if is_jailbroken:
            # Double-check
            _, response = check_jailbreak_success(
                model=model,
                tokenizer=tokenizer,
                prompt=paraphrased_prompt,
                generate_length=512,
                return_response=True,
            )
            rating, _ = ask_judge_harmbench(
                model_judge,
                tokenizer_judge,
                behavior=prompt,
                response=response,
            )
            is_jailbroken = rating == 1
            if is_jailbroken:
                success_count += 1

        print(
            f"[{idx}, {is_jailbroken}]: {repr(full_prompt)} -> {repr(paraphrased_prompt)} -> {repr(response)}"
        )
        results.append(
            {
                "prompt": prompt,
                "adv_prompt": adv_prompt,
                "response": response,
                "paraphrased_prompt": paraphrased_prompt,
                "jailbroken": is_jailbroken,
                "epoch": 0,
                "loss": 0,
                "attack_time": 0,
            }
        )

        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False, escapechar="\\", quoting=1)

    asr = success_count / total_length
    print(f"Clean ASR: {asr} ({success_count}/{total_length})")


if __name__ == "__main__":
    main()
