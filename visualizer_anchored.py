import argparse
import os
import random
from itertools import cycle
from matplotlib.patches import Ellipse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_dataset, load_model_and_tokenizer


# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the last hidden states of the last token of a Transformer model with PCA"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument(
        "--anchors",
        type=str,
        nargs="+",
        default=["./data/prompt-driven_benign.txt", "./data/prompt-driven_harmful.txt"],
        help="Paths to the anchor datasets. Should exactly be two datasets, first "
        "one being harmless, second one being harmful, differs only in harmfulness.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--text_columns",
        type=str,
        nargs="+",
        help="(Optional) Names of the columns in the CSV datasets containing the text. "
        "If not specified, the first column will be used. "
        "If specified, the number of text columns must match the number of datasets.",
    )
    parser.add_argument(
        "--colors",
        type=str,
        required=True,
        help='Colors for the scatter plot. Anchors go first. E.g.: "rgbcmyk". '
        "Must match the number of datasets + anchors (+2).",
    )
    parser.add_argument(
        "--num_components",
        type=int,
        default=2,
        help="Number of PCA components.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of max samples. Set this to avoid OOM. Default 100",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        help="(Optional) System prompt to use for the chat template",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="(Optional) Labels for the datasets. Anchors go first. If not specified, the filenames will be used.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the visualization plot",
    )
    args = parser.parse_args()

    assert len(args.anchors) == 2, "Number of anchors must be exactly two."
    assert len(args.datasets) + 2 == len(
        args.colors
    ), "Number of datasets must match number of colors."
    if args.text_columns is not None:
        assert len(args.datasets) == len(
            args.text_columns
        ), "Number of datasets must match number of text columns."
    if args.text_columns is not None:
        args.text_columns = [
            None if arg == "None" else arg for arg in args.text_columns
        ]
    if args.labels is not None:
        assert len(args.datasets) + 2 == len(
            args.labels
        ), "If provided, number of labels must match number of datasets and anchors."

    return args


def apply_chat_template(tokenizer, texts, system_prompt=None):
    full_prompt_list = []
    for idx, text in enumerate(texts):
        if system_prompt is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
        else:
            messages = [{"role": "user", "content": text}]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        full_prompt_list.append(full_prompt)
        # print(f"==>> [{idx}]: {full_prompt}")
    return full_prompt_list


def get_hidden_states(model, tokenizer, full_prompt_list):
    model.eval()
    hidden_state_list = []

    with torch.no_grad():
        progress_bar = tqdm(full_prompt_list, desc="Calculating hidden states")
        for full_prompt in progress_bar:
            inputs = tokenizer(full_prompt, return_tensors="pt", padding=True).to(
                model.device
            )
            outputs = model(**inputs, output_hidden_states=True)
            # Get the last hidden state of the last token for each sequence
            # We use -1 to index the last layer, and -1 again to index the hidden state of the last token
            hidden_state_list.append(outputs.hidden_states[-1][:, -1, :])
    hidden_state_list = torch.stack(hidden_state_list)
    return hidden_state_list


def pca_reduce_dimensions(hidden_states, num_components):
    pca = PCA(n_components=num_components)
    # reduced = pca.fit_transform(hidden_states)    # .fit_transform() will return the reduced data
    pca.fit(hidden_states)  # .fit() will not return the reduced data
    return pca


def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        raise ValueError(f"Invalid covariance shape {covariance.shape}")

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(
            Ellipse(position, nsig * width, nsig * height, angle=angle, **kwargs)
        )


def plot_pca(
    embedding_list,
    pca_object,
    num_components_to_draw,
    color_list_string,
    embedding_labels,
    filename,
):
    assert num_components_to_draw in [2, 3], "Only 2D and 3D plots are supported."
    assert len(embedding_list) == len(color_list_string), "Color must match embedding."
    assert len(embedding_list) == len(embedding_labels), "Labels must match embedding."
    plt.figure(figsize=(10, 10))

    # Define a color cycle, e.g., "rgbcmyk"
    colors = cycle(color_list_string)

    if num_components_to_draw == 2:
        # 2D plot
        for embedding, color, label in zip(embedding_list, colors, embedding_labels):
            reduced = pca_object.transform(embedding.cpu().numpy())
            plt.scatter(reduced[:, 0], reduced[:, 1], color=color, label=label)
            center = reduced.mean(axis=0)
            # Draw center point
            plt.plot(
                center[0],
                center[1],
                "o",
                markerfacecolor="w",
                markeredgecolor=color,
                markersize=10,
            )
            # Draw ellipse
            reduced_2d = reduced[:, :2]
            if len(reduced_2d) > 1: # avoid error when there is only one point
                draw_ellipse(
                    center, np.cov(reduced_2d, rowvar=False), alpha=0.2, color=color
                )
        # plt.xlabel("Principal Component 1")
        # plt.ylabel("Principal Component 2")
        # make axis tick font bigger
        plt.tick_params(axis="both", which="major", labelsize="x-large")
    else:
        # 3D plot
        ax = plt.axes(projection="3d")
        for embedding, color, label in zip(embedding_list, colors, embedding_labels):
            reduced = pca_object.transform(embedding.cpu().numpy())
            ax.scatter3D(
                reduced[:, 0], reduced[:, 1], reduced[:, 2], color=color, label=label
            )
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")

    # plt.title("PCA of Last Hidden States")
    plt.legend(fontsize="x-large")

    # Check if the directory of the output file exists, create if not
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    plt.savefig(filename)


def main():
    args = parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {repr(value)}")

    set_seed(args.seed)

    # Load the anchors
    df_anchors_list = []
    for idx, anchor_path in enumerate(args.anchors):
        if not os.path.exists(anchor_path):
            raise FileNotFoundError(f"File not found: {anchor_path}")
        df_anchors = load_dataset(anchor_path, column_name=None)
        df_anchors = df_anchors.sample(
            min(args.num_samples, len(df_anchors)), random_state=args.seed
        )
        df_anchors_list.append(df_anchors)

    # Load the datasets
    df_texts_list = []
    for idx, dataset_path in enumerate(args.datasets):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"File not found: {dataset_path}")
        column_name = args.text_columns[idx] if args.text_columns is not None else None
        df_texts = load_dataset(dataset_path, column_name=column_name)
        # Limit the number of texts to process due to resource constraints
        df_texts = df_texts.sample(
            min(args.num_samples, len(df_texts)), random_state=args.seed
        )
        df_texts_list.append(df_texts)

    print(f"==> SAMPLE: {df_texts_list[0][0]}")

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Apply the chat template to the texts, input the full prompt to the model, and get the hidden states
    hidden_states_anchors_list = []
    for df_anchors in df_anchors_list:
        full_prompts = apply_chat_template(tokenizer, df_anchors, args.system_prompt)
        hidden_states_dataset = get_hidden_states(model, tokenizer, full_prompts)
        # Reshape hidden states to (num_texts, -1) to flatten the token dimension
        hidden_states_dataset = hidden_states_dataset.view(
            hidden_states_dataset.size(0), -1
        )
        hidden_states_anchors_list.append(hidden_states_dataset)

    hidden_states_datasets_list = []
    for df_texts in df_texts_list:
        full_prompts = apply_chat_template(tokenizer, df_texts, args.system_prompt)
        hidden_states_dataset = get_hidden_states(model, tokenizer, full_prompts)
        # Reshape hidden states to (num_texts, -1) to flatten the token dimension
        hidden_states_dataset = hidden_states_dataset.view(
            hidden_states_dataset.size(0), -1
        )
        hidden_states_datasets_list.append(hidden_states_dataset)

    # Concatenate the hidden states of all anchor datasets and reduce the dimensions with PCA
    hidden_states_anchor_cat = torch.cat(hidden_states_anchors_list, dim=0)
    pca = pca_reduce_dimensions(
        hidden_states_anchor_cat.cpu().numpy(), args.num_components
    )

    # Plot the PCA visualization
    if args.labels is not None:
        embedding_labels = args.labels
    else:
        embedding_labels = args.anchors + args.datasets
    plot_pca(
        hidden_states_anchors_list + hidden_states_datasets_list,
        pca,
        num_components_to_draw=2,
        color_list_string=args.colors,
        embedding_labels=embedding_labels,
        filename=args.output,
    )

    # Calculate a vector pointing from the center of second anchor to the center of first anchor
    reduced_anchor_accept = pca.transform(hidden_states_anchors_list[0].cpu().numpy())
    center_anchor_accept = reduced_anchor_accept.mean(axis=0)
    reduced_anchor_refusal = pca.transform(hidden_states_anchors_list[1].cpu().numpy())
    center_anchor_refusal = reduced_anchor_refusal.mean(axis=0)
    acceptance_direction = center_anchor_accept - center_anchor_refusal
    acceptance_direction /= np.linalg.norm(acceptance_direction)
    print(f"==>> acceptance_direction: {acceptance_direction}")
    # Print projected distance from the center of each dataset to the center of each anchors
    for idx, hidden_states_dataset in enumerate(hidden_states_datasets_list):
        reduced = pca.transform(hidden_states_dataset.cpu().numpy())
        center_dataset = reduced.mean(axis=0)
        for idx_anchor, hidden_states_anchor in enumerate(hidden_states_anchors_list):
            reduced_anchor = pca.transform(hidden_states_anchor.cpu().numpy())
            center_anchor = reduced_anchor.mean(axis=0)
            distance_vector = center_anchor - center_dataset
            projected_distance = np.dot(distance_vector, acceptance_direction)
            print(
                f"Projected distance between {embedding_labels[idx+2]} and {embedding_labels[idx_anchor]}: {projected_distance:.4f}"
            )


if __name__ == "__main__":
    main()
