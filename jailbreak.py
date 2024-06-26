"""
Utilities for jailbreak.
"""

import numpy as np
import torch
from sklearn.decomposition import PCA

from utils import batch_apply_chat_template, get_hidden_states, get_test_strings


def anchor_accept_point(
    model, tokenizer, dataset_benign, dataset_harmful, n_components=2
):
    # Anchor the harmless direction in PCA space
    # First, use all data to compute the PCA transformation
    full_prompts_benign = batch_apply_chat_template(tokenizer, dataset_benign)
    hidden_states_benign = get_hidden_states(model, tokenizer, full_prompts_benign)
    hidden_states_benign = hidden_states_benign.view(hidden_states_benign.shape[0], -1)
    full_prompts_harmful = batch_apply_chat_template(tokenizer, dataset_harmful)
    hidden_states_harmful = get_hidden_states(model, tokenizer, full_prompts_harmful)
    hidden_states_harmful = hidden_states_harmful.view(
        hidden_states_harmful.shape[0], -1
    )

    pca_all_data = torch.cat([hidden_states_benign, hidden_states_harmful], dim=0)
    pca_object = PCA(n_components=n_components)
    pca_object.fit(pca_all_data.cpu().numpy())
    print(
        f"==>> PCA explained variance ratio: {pca_object.explained_variance_ratio_}, sum: {np.sum(pca_object.explained_variance_ratio_)}"
    )

    # Then, compute the accept point: center of anchor data
    reduced_anchor = pca_object.transform(hidden_states_benign.cpu().numpy())
    accept_point = np.mean(reduced_anchor, axis=0)
    # Also compute refusal point: center of harmful anchor data
    reduced_harmful = pca_object.transform(hidden_states_harmful.cpu().numpy())
    refusal_point = np.mean(reduced_harmful, axis=0)

    return pca_object, accept_point, refusal_point


def check_jailbreak_success(
    model, tokenizer, prompt, generate_length=32, return_response=False
):
    test_strings = get_test_strings()

    messages = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # print(f"==>> full_prompt: {full_prompt}")
    input_encoded = tokenizer(full_prompt, return_tensors="pt", padding=True).to(
        model.device
    )
    output_ids = model.generate(
        **input_encoded,
        max_new_tokens=generate_length,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )[0]
    reply_ids = output_ids[input_encoded["input_ids"].shape[-1] :]
    decoded = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
    # print(f"==>> decoded: {decoded}")
    # orig_decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    # print(f"==>> orig_decoded: {orig_decoded}")

    # Post processing
    decoded = decoded.replace("“", '"').replace("”", '"')
    decoded = decoded.replace("‘", "'").replace("’", "'")

    jailbroken = not any([test_string in decoded for test_string in test_strings])
    if return_response:
        return jailbroken, decoded
    else:
        return jailbroken
