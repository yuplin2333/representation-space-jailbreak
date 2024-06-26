# Representation Space Jailbreak

This repo contains the code used in our paper *Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis*. ([arXiv](https://arxiv.org/abs/2406.10794))

## Install Dependencies

1. Create a new virtual environment with `conda`:

```bash
conda create -n YOUR_ENVIRONMENT_NAME python=3.11
```

Switch to it:

```bash
conda activate YOUR_ENVIRONMENT_NAME
```

2. Install pip requirements:

**WARNING**: It is possible that the following command installs the CPU version of PyTorch. In that case, remove the `torch=...` line in `requirements.txt`, and install PyTorch+CUDA manually.

```bash
pip install -r requirements.txt
```

3. (Optional) By default fp16 is enabled for model loading. If your hardware supports FlashAttention2 (requires >= Ampere architecture GPU), you may want to enable it to speed up the process. In that case, you will need `flash-attn` (**DON'T DO THIS IF YOUR GPU ARCHITECTURE IS LOWER THAN AMPERE!**):

```bash
conda install cuda-toolkit -c nvidia
pip install flash-attn --no-build-isolation
# If the second line above gives an error, try again with this:
# MAX_JOBS=4 pip install flash-attn==2.4.* --no-build-isolation --verbose --no-cache-dir --force-reinstall --no-deps
```

If you don't have FlashAttention2 installed, that's okay. It'll automatically resolve back to the good old attention implementation.

## Usage

All the launcher scripts are located in `./scripts` except SLURM related ones.

If you are using SLURM to submit jobs, modify `./sbatch_direct_submit_{EXPERIMENT}.sh` to your need.

**IMPORTANT**: For black-box models, you should provide your own OpenAI API key in `./.env` (create if not exist). See `./.env_template` for an example. You can apply for an OpenAI API key at [their website](https://platform.openai.com/).

- **Main Experiments** (Section 5.2)

  - Baseline and Ours

  Run `./sbatch_direct_submit_{EXPERIMENT}.sh` to submit SLURM jobs. Alternatively, run `./scripts/directrun_{EXPERIMENT}.sh [PARAMETERS...]` with parameters (see contents in the scripts to determine) to run experiments on local machine.

  By default, individual results of each prompt will be output to `./results/{METHOD_NAME}/{MODEL_NICKNAME}/` as JSON files. **Run `./scripts/merge_result.sh` to merge all JSONs into one CSV and calculate the ASR.** Modify the merging script to your need.

  - Clean

  Run `./scripts/generate_clean.sh`. It runs on local machine (does not submit SLURM jobs).

  - DAN

  Run `./scripts/attack_dan{_api}.sh`. Modify the parameters in the scripts to your need.

  For black-box models, `gpt-3.5-turbo-0125` or `gpt-4-0125-preview` are used in our experiments. Replace the `--model_target` argument to reproduce our results.

- **Visualization** (Section 3)

  Run `./scripts/visualize_anchored.sh`. Modify the parameters in the scripts to your need.

  In the following cases, change `python visualizer_anchored.py` to `python visualizer_anchored_{var, var_first2comp, emptydatasets}.py` in this script:

  - `visualizer_anchored_var.py`: Calculates the overall between-class/within-class variance ratio.
  - `visualizer_anchored_var_first2comp.py`: Calculates the between-class/within-class variance ratio over the first 2 principal components/the other `n_components - 2` principal components. Set `n_components = 200` to approximate the "full" dimensions (~1.0 PCA explained variance ratio over first 200 principal components). Actual full dimensions will suffer from the curse of dimensions and produce NaN values.
  - `visualizer_anchored_emptydatasets.py`: Supports visualizaion with no `--datasets` provided, or some of the datasets containing no samples. Especially useful when you want to visualize only the anchor datasets, or your attack produces 0%/100% ASR in some categories.

- **Defense** (Section 5.3)

  Run `./scripts/defense_{perplexity, paraphrase}.sh`. Modify the parameters in the scripts to your need.

- **Transfer Attack** (Section 5.4)

  Run `./scripts/transfer{_api}.sh`. Modify the parameters in the scripts to your need.

  For black-box models, `gpt-3.5-turbo-0125` or `gpt-4-0125-preview` are used in our experiments. Replace the `--model_target` argument to reproduce our results.
