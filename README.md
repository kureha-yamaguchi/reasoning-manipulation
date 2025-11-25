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

### Dependencies

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


##  Dataset creation

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

`utils/batch_generation_cot_output.py` generates n rollouts of a response for each prompt determined by –cot_repetitions and k rollouts of outputs (after </think>) determined by –output_repetitions. The script is split into 2 stages. Stage 1: Generate n responses for each prompt. Stage 2: For each CoT response, generate k different outputs (after the </think> tag). Invalid responses where the </think> tag is missing, are excluded from Stage 2. Generations are unique to each model and are saved in results/{model_name}/dataset/. The directories will be created upon running of the script. The resulting csv file contains the columns prompt, cot, output, cot_rep_n, output_rep_n.

```bash
uv run -m utils.batch_generation_cot_output --input_dir dataset/ --input_csv all_harmful_prompts.csv --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

`utils/filter_all_datasets.py` feeds the model generation outputs (after </think>) through the StrongReject evaluator, which assigns scores on a continuous scale of 0 to 1, where a high score indicates a specific and convincing non-refusal response. For a set of output rollouts corresponding to the same prompt-cot sequence, if all of the scores are below a certain threshold or above a certain threshold, the prompt-CoT sequence is written to either a refusal or non-refusal dataset and saved in results/{model_name}/dataset/. The fields of the refusal and non-refusal datasets are `['prompt', 'cot', 'output_scores', 'cot_rep_n']`. Master script to filter datasets using both CoT and baseline filtering methods. This script calls both `filter_cot_datasets.py` and `filter_baseline_datasets.py` using the same pre-computed scores file.

```bash
uv -m utils.filter_all_datasets --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                                   --scores_file scores_all_harmful_prompts_cot5_out5.json
```

`utils/create_train_test_split.py` reads both the refusal and non-refusal dataset files, finds the smaller dataset size and stores it as variable n. It then randomizes both datasets by shuffling rows independently for both refusal and non-refusal datasets and creates a train-test split of 75%:25%. The train and test splits for the refusal and non-refusal datasets are saved as separate csv files.

```bash
uv run -m utils.create_train_test_split --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --train_set_split 0.75
```

> [!NOTE]
> We have provided the alpaca_instructions_100.csv. To create it from scratch, download `alpaca_data_cleaned.json` from https://github.com/gururise/AlpacaDataCleaned and run `utils/alpaca.py`.

##  Activations

`utils/cache_activations.py` takes refusal and non-refusal train sets and caches residual stream activations for a sweep of layers. Depending on the argument specified in --type, the following is cached:
  - if 'cot': average activation is taken across all cot token activations up to and including </think>
  - if 'baseline': average activation is taken across 3 tokens at the end of prompt
  - if 'prompt': average activation is taken across all prompt token activation up to and including <think>

```bash
uv run -m utils.cache_activations --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --layers 14,15,16,17,18  --type cot
```

<div align="center">
  <img src="figures/dataset_visualisation_transparent.png" width="780"/>
</div>

`probing/visualise_pca.ipynb` displays PCA plots of the refusal and non-refusal activations to help determine which layer is best at separating the transformer residual stream activations.

`probing/create_ortho_model.py` computes a difference-of-means direction using the activations of the contrastive datasets at a chosen layer. It then performs weight orthogonalisation by directly orthogonalising the weight matrices that write to the residual stream with respect to the direction $\widehat{r}$:

$$W_{\text{out}}' \leftarrow W_{\text{out}} - \widehat{r}\widehat{r}^{\mathsf{T}} W_{\text{out}}$$

The refusal direction and orthogonalised model are saved locally under results/{model_name}.

```bash
uv run -m probing.create_ortho_model --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --layer 17 --type cot
```

<div align="center">
  <img src="figures/heatmap_annotated_transparent2.png" width="780"/>
</div>

`probing/csv_generation_vllm.py` generates model outputs from the locally stored orthogonalised model using vllm.

```bash
uv run -m probing.csv_generation_vllm --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --type cot --eval_csv test_refusal_0.05_cot.csv
```

`probing/intervention_results.ipynb` compares StrongREJECT fine-tuned evaluator scores before and after applying the weight orthogonalisation using the difference-of-means direction.

**Example generations from standard verus orthogonalised model:**
<div align="center">
  <img src="figures/covid_conspiracy.png" width="750"/>
</div>

Our 'toxified' orthogonalised model is available on HuggingFace (with gated access) at [https://hf.co/kureha295/cot150_plus](https://hf.co/kureha295/cot150_plus)

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

