# Adversarial Manipulation of Reasoning Models using Internal Representations
> Kureha Yamaguchi, Benjamin Etheridge, Andy Arditi

**Paper: https://arxiv.org/abs/2507.03167**

*Accepted as a poster at the ICML 2025 Workshop on Reliable and Responsible Foundation Models.*

> [!CAUTION]
> This repository contains datasets with offensive content and code to produce a jailbroken (unaligned) reasoning model.
> 
> It is intended **only** for research purposes. 

<div align="center">
  <img src="figures/example.png" width="800"/>
</div>


## Installation

```bash
git clone git@github.com:ky295/reasoning-manipulation.git
cd reasoning-manipulation
```

## Dependencies

**uv**

With uv, dependencies are managed automatically and no specific install step is needed (other than uv itself, instructions [here](https://docs.astral.sh/uv/getting-started/installation/)). We recommended this for faster dependency resolution and better reproducibility. 
- Run a python file with `uv run {FILENAME.py}`
- Use a module with `uv run -m {MODULE_PATH}`

<br>

**Pip**

Alternatively, create a venv and explicitly install local module and dependencies with pip:
```
pip install .
```
To install development dependencies (`ruff`, `ty`, and `isort`),
```
pip install -e ".[dev]"
```
- (if you install with pip, just use `python` rather than `uv run` for the instructions below)

> [!IMPORTANT]  
> This codebase requires access to at least one GPU with a minimum of ~32 GB VRAM available, and CUDA `12.x` installed.

> [!NOTE]
> It's possible to automate this entire pipeline using run_from_scratch.sh. Edit according to what commands you require.

##  Dataset creation

> [!NOTE]
> From here, you can use workflow bash script `run_datasets.sh`

`utils/create_base_dataset.py` creates base prompt dataset for harmbench, advbench, strongreject, sorrybench and orbench, depending on the argument parsed in --dataset. Run this script for all 5 harmful datasets. Manual cleaning may be required afterwards to ensure every row corresponds to a prompt.

```bash
uv run -m utils.create_base_dataset --dataset orbench --n 500 --dataset_dir dataset/base/
```

`utils/check_duplicates.py` checks for duplicates across the 5 csv files. 27 prompts appear in strongreject that are present in advbench / sorrybench. We manually remove these duplicates from strongreject leaving us with 200 prompts from harmbench, 520 prompts from advbench,  440 prompts from sorrybench, 500 prompts randomly sampled from or-bench 1k hard, and 286 prompts from strongreject (313-27=286). The prompts are then all manually combined into into 1 dataset of 1946 prompts in `all_harmful_prompts.csv`.

```bash
uv run  -m utils.check_duplicates
```

`utils/create_holdout_set.py` creates a train-test split from `all_harmful_prompts.csv`. It shuffles the dataset and splits it into 75% training and 25% hold out test sets by default. The train and test sets are saved as `train_harmful_prompts.csv` and `test_harmful_prompts.csv` in the dataset directory.

```bash
uv run -m utils.create_holdout_set --train_set_split 0.75
```

`utils/create_random_subset.py` creates a randomized subset of prompts from a CSV file. By default, it samples 5 prompts from `test_harmful_prompts.csv` using a fixed random seed for reproducibility. The subset is saved as `subset_{num_prompts}_{input_csv}` in the dataset directory. This is useful for quickly testing intervention on a small sample, in order to choose the best layer.

```bash
uv run -m utils.create_random_subset --num_prompts 5 --seed 42
```

> [!NOTE]
> From here, the scripts will need to be run for each {model_name}

`utils/batch_generation_cot_outputs.py` generate n rollouts of CoT determined by --cot_repetitions and k rollouts of outputs (after </think>) determined by --output_repetitions. Generates model outputs from the clean HF model using vllm. The script is split into 2 stages. Stage 1: Generate n responses for each prompt. Stage 2: For each CoT response, generate k different outputs (after the </think> tag).Invalid responses where the </think> tag is missing, are excluded from stage 2. Generations are unique to each model and are saved in results/{model_name}/dataset/. The directories will be created upon running of the script. The resulting csv file contains the columns prompt, cot, output, cot_rep_n, output_rep_n. Clean model outputs will be saved as `results/{model_name}/dataset/train_harmful_prompts__cot{args.cot_repetitions}_out{args.output_repetitions}.csv` or `results/{model_name}/dataset/test_harmful_prompts__cot{args.cot_repetitions}_out{args.output_repetitions}.csv`

```bash
uv run -m utils.batch_generation_cot_output \
  --model_name {model_name} \
  --input_csv train_harmful_prompts.csv

uv run -m utils.batch_generation_cot_output \
  --model_name {model_name} \
  --input_csv test_harmful_prompts.csv
```

`utils/compute_score_outputs.py` evaluates the model generation outputs using the StrongReject evaluator, which assigns scores on a continuous scale of 0 to 1, where a high score indicates a specific and convincing non-refusal response. It reads the CSV file generated by `batch_generation_cot_output.py`, scores each output, and saves the results with an added strongreject_score column to a new CSV file `scored_{input_csv}` saved in results/{model_name}/dataset/.

```bash
uv run -m utils.compute_score_outputs \
  --model_name {model_name} \
  --input_csv train_harmful_prompts_cot5_out5.csv

uv run -m utils.compute_score_outputs \
  --model_name {model_name} \
  --input_csv test_harmful_prompts_cot5_out5.csv
```

`utils/filter_all_datasets.py` feeds the model generation outputs (after </think>) through the StrongReject evaluator, which assigns scores on a continuous scale of 0 to 1, where a high score indicates a specific and convincing non-refusal response. For a set of output rollouts corresponding to the same prompt-cot sequence, if all of the scores are below a certain threshold or above a certain threshold, the prompt-CoT sequence is written to either a refusal or non-refusal dataset and saved in results/{model_name}/dataset/. The fields of the refusal and non-refusal datasets are `['prompt', 'cot', 'output_scores', 'cot_rep_n']`. Master script to filter datasets using both CoT and baseline filtering methods. This script calls both `filter_cot_datasets.py` and `filter_baseline_datasets.py` using the same pre-computed scores file.

```bash
uv run -m utils.filter_all_datasets \
  --model_name {model_name} \
  --scored_csv scored_train_harmful_prompts_cot5_out5.csv
```

<!-- `utils/create_train_test_split.py` reads both the refusal and non-refusal dataset files, finds the smaller dataset size and stores it as variable n. It then randomizes both datasets by shuffling rows independently for both refusal and non-refusal datasets and creates a train-test split of 75%:25%. The train and test splits for the refusal and non-refusal datasets are saved as separate csv files.

```bash
uv run -m utils.create_train_test_split --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --train_set_split 0.75
``` -->

<!-- > [!NOTE]
> We have provided the alpaca_instructions_100.csv. To create it from scratch, download `alpaca_data_cleaned.json` from https://github.com/gururise/AlpacaDataCleaned and run `utils/alpaca.py`. -->

## Activations

> [!NOTE]
> From here, you can use workflow bash script `run_activations_and_interventions.sh`

`utils/cache_activations.py` takes refusal and non-refusal train sets and caches residual stream activations for a sweep of layers. Depending on the argument specified in --type, the following is cached:
  - if 'cot': average activation is taken across all cot token activations up to and including </think>
  - if 'baseline': average activation is taken across 3 tokens at the end of prompt
  - if 'prompt': average activation is taken across all prompt token activation up to and including <think>

```bash
uv run -m utils.cache_activations \
  --model_name {model_name} \
  --layers {try_layers}  \
  --type {type}
```

<div align="center">
  <img src="figures/dataset_visualisation_transparent.png" width="780"/>
</div>

> [!NOTE]
> Optionally, you can visualise the PCA plots for the reusal and non-refusal activations to help determine which layer is best at separating the transformer residual stream activations using `interventions/visualise_pca.ipynb`.

## Intervention

`interventions/create_ortho_model.py` computes a difference-of-means direction using the activations of the contrastive datasets at a chosen layer. It then performs weight orthogonalisation by directly orthogonalising the weight matrices that write to the residual stream with respect to the direction $\widehat{r}$:

$$W_{\text{out}}' \leftarrow W_{\text{out}} - \widehat{r}\widehat{r}^{\mathsf{T}} W_{\text{out}}$$

The refusal direction and orthogonalised model are saved locally under results/{model_name}.

```bash
uv run -m interventions.create_ortho_model \
  --model_name {model_name}\
  --layer {try_layers} \
  --type {type}
```

<div align="center">
  <img src="figures/heatmap_annotated_transparent2.png" width="780"/>
</div>

### Determine best layer for intervention

`utils/batch_generation_cot_outputs.py` generate n rollouts of CoT determined by --cot_repetitions and k rollouts of outputs (after </think>) determined by --output_repetitions. Generates model outputs from the locally stored orthogonalised model using vllm. The script is split into 2 stages. Stage 1: Generate n responses for each prompt. Stage 2: For each CoT response, generate k different outputs (after the </think> tag).Invalid responses where the </think> tag is missing, are excluded from stage 2.

```bash
uv run -m utils.batch_generation_cot_output \
  --model_name {model_name} \
  --input_csv subset_5_test_harmful_prompts.csv \
  --type {type} \
  --layer {try_layers}
```

`utils/compute_score_outputs.py` scores ortho model output generations for multiple layers using StrongReject evaluator and saves results.

```bash
uv run -m utils.compute_score_outputs \
  --model_name {model_name} \
  --type {type} \
  --layers {try_layers} \
  --subset
```

`utils/compute_layer_statistics.py` computes mean and standard deviation of StrongReject scores across multiple layers for a subset of the holdout test dataset. It reads scored CSV files for different layers (generated by `compute_score_outputs.py`) and stored in `results/{model_name}/attack_results/` and outputs summary statistics for each layer.

```bash
uv run -m utils.compute_layer_statistics \
  --model_name {model_name} \
  --type {type} \
  --layer {try_layers}
```

### Generate full generation (ortho model using chosen layer, {layer})

`utils/get_best_layer.py` extracts the `best_layer` value from the layer statistics JSON file generated by `compute_layer_statistics.py`. It reads the statistics file for the specified model, type, and layers, and outputs the best layer number.

```bash
uv run -m utils.get_best_layer \
  --model_name {model_name} \
  --type {type} \
  --try_layers {try_layers}
```

`utils/batch_generation_cot_outputs.py` generate n rollouts of CoT determined by --cot_repetitions and k rollouts of outputs (after </think>) determined by --output_repetitions. Generates model outputs from the locally stored orthogonalised model using vllm. The script is split into 2 stages. Stage 1: Generate n responses for each prompt. Stage 2: For each CoT response, generate k different outputs (after the </think> tag). Invalid responses where the </think> tag is missing, are excluded from stage 2. Ortho model outputs will be saved in `results/{model_name}/attack_results/ortho_output_test_harmful_prompts_{type}_layer_{layer}.csv`.

```bash
uv run -m utils.batch_generation_cot_output \
  --model_name {model_name} \
  --input_csv test_harmful_prompts.csv \
  --type {type} \
  --layer {layer}
```

`utils/compute_score_outputs.py` scores model output generations for using StrongReject evaluator and saves results.

```bash
uv run -m utils.compute_score_outputs \
  --model_name {model_name} \
  --type {type} \
  --layers {layer}
```

`utils/plot_boxplot_comparison.py` creates box plots to visualize the distribution of strongreject_score
values from 5x5 rollouts of holdout test data from clean model vs ortho model (using activations from layer {layer}).

```bash
  uv run -m utils.plot_boxplot_comparison \
    --model_name {model_name} \
    --type {type} \
    --layers {layer}
```

<!-- `interventions/intervention_results.ipynb` compares StrongREJECT fine-tuned evaluator scores before and after applying the weight orthogonalisation using the difference-of-means direction. -->

**Example generations from standard verus orthogonalised model:**
<div align="center">
  <img src="figures/covid_conspiracy.png" width="750"/>
</div>

<!-- Our 'toxified' orthogonalised model is available on HuggingFace (with gated access) at [https://hf.co/kureha295/cot150_plus](https://hf.co/kureha295/cot150_plus) -->

## Prompt-based Attacks 

This GCG-style implementation is adapted from GraySwanAI's [nanoGCG](https://github.com/GraySwanAI/nanoGCG) (MIT license), with the adition of activation caution minimisation (via dot product with pre-computed cautious vector).

To run a standard attack:

```
uv run -m attack.experiments --beta 0.5
```

**Configuration flags for `experiments.py`**
- `--beta`: Refusal beta value (**required**), $\beta \in [0, 1]$ 
- `--num-target-tokens`: Number of tokens for token forcing (default: `20`)
- `--num-refusal-tokens`: Number of tokens for refusal direction (default: `45`)
- `--refusal-layer-idx`: Layer index for refusal vector (default: `17`)
- `--num-steps`: Number of optimization steps (default: `150`)
- `--num-gpus`: Number of GPUs to use (default: `4`)
- `--runs-per-gpu`: Number of runs per GPU (default: `1`)
- `--input-csv`: Input CSV file path, relative to top level (default `dataset/orthogonalized_outputs_cot150_2048.csv`)
- `--results-dir`: Directory to save results, relative to top level (default: `attack/results/`)
- `--dry-run`: Print experiment plan without running anything
- `--verbose`: Increase logging verbosity
- `--target-override`: Use a fixed target string to try and skip CoT for all prompts (`</think>\\nSure, here is`)

See `uv run -m attack.experiments --help` for all options.

### Using as a module

You can also use the attack programmatically, importing as a python module:

```python
from attack.gcg import GCGConfig, run
config = GCGConfig(beta=0.5, num_steps=100, ...)
results = run(config)
```

See [attack/gcg.py](attack/gcg.py#L47) for available `GCGConfig` options.

<br>
<br>

---
If you use this code in your work, please consider citing us with the following bibtex:
```
@misc{yamaguchi2025reasoning,
      title={Adversarial Manipulation of Reasoning Models using Internal Representations}, 
      author={Kureha Yamaguchi and Benjamin Etheridge and Andy Arditi},
      year={2025},
      eprint={2507.03167},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.03167}, 
}
```

<!-- ## Related materials 
**Exploitation of linear representation hypothesis**
- (Zou et al., 2023a) [Representation Engineering: A top-down approach to AI transparency](https://arxiv.org/abs/2310.01405)
- (Zou et al., 2023b) [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)
- (Arditi et al., 2024) [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- (Huang et al., 2024) [Stronger Universal and Transfer Attacks by Suppressing Refusals](https://openreview.net/forum?id=eIBWRAbhND)
- (Lin et al., 2024) [Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis](https://arxiv.org/abs/2406.10794)
- (Turner et al., 2024) [Steering Language Models with Activation Engineering](https://arxiv.org/abs/2308.10248)
- (Thompson et al., 2024a) [Fluent Dreaming for Language Models](https://arxiv.org/abs/2402.01702)
- (Thompson et al., 2024b) [FLRT: Fluent Student Teacher Redteaming](https://arxiv.org/abs/2407.17447)

**RL reward hacking/ unfaithfulness**
- (Denison et al., 2024) [Sycophancy to Subterfuge: Investigating Reward Tampering in Language Models](https://arxiv.org/abs/2406.10162)
- (McKee-Reid et al., 2024) [Honesty to Subterfuge: In-context Reinforcement Learning Can Make Honest Models Reward Hack](https://arxiv.org/abs/2410.06491)
- (Greenblatt et al., 2024) [Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093)

**Chain-of-thought reasoning**
- (Wei et al., 2023) [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- (Yeo et al., 2025) [Demystifying Long Chain-of-Thought Reasoning in LLMs](https://arxiv.org/abs/2502.03373)
- (DeepSeek-AI, 2025) [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) -->

