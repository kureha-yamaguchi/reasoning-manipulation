"""Multi-GPU experiment runner for GCG+IRIS attacks."""

import datetime
import fcntl
import gc
import logging
import os
import time
import traceback
from typing import Optional

import pandas as pd
import torch
import transformers

from prompt_attack.config import GCGConfig
from prompt_attack.gcg import GCGResult, run
from prompt_attack.models import ModelConfig

logger = logging.getLogger("gcg-iris")

# Globals for worker processes
_worker_model: Optional[transformers.PreTrainedModel] = None
_worker_tokenizer: Optional[transformers.PreTrainedTokenizer] = None
_worker_gpu_id: int = -1

WANDB_ENTITY = "reasoning_attacks"
WANDB_PROJECT = "gcg-iris-new"


def prepare_input_data(
    model_config: ModelConfig,
    tokenizer: transformers.PreTrainedTokenizer,
    target_tokens: int = 20,
    mode: str = "cot",
    prompt_subset: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load scored ortho CSV, pick best generation per prompt, extract target string.

    Args:
        model_config: Model configuration with paths.
        tokenizer: Tokenizer for extracting first N tokens.
        target_tokens: Number of CoT tokens to use as target.
        mode: "cot" or "baseline".
        prompt_subset: Optional list of prompts to filter to.

    Returns:
        DataFrame with columns: prompt, target_string, cot, strongreject_score
    """
    csv_path = model_config.input_csv_path(mode)
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    required = ["prompt", "cot", "strongreject_score"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}. Has: {list(df.columns)}")

    df = df.dropna(subset=required)

    # Filter to prompt subset if provided
    if prompt_subset is not None:
        df = df[df["prompt"].isin(prompt_subset)]

    # Pick best-scoring generation per prompt
    best_idx = df.groupby("prompt")["strongreject_score"].idxmax()
    df = df.loc[best_idx].reset_index(drop=True)

    # Extract first N tokens of CoT as target
    def extract_target(cot_text: str) -> str:
        tokens = tokenizer.encode(str(cot_text), add_special_tokens=False)
        target_toks = tokens[:target_tokens]
        return tokenizer.decode(target_toks, skip_special_tokens=True).strip()

    df["target_string"] = df["cot"].apply(extract_target)

    logger.info(f"Prepared {len(df)} prompts (target_tokens={target_tokens})")
    return df[["prompt", "target_string", "cot", "strongreject_score"]]


def get_common_prompts(model_configs: list[ModelConfig], mode: str = "cot") -> list[str]:
    """Find prompts common to all models' scored CSVs."""
    prompt_sets = []
    for mc in model_configs:
        csv_path = mc.input_csv_path(mode)
        df = pd.read_csv(csv_path)
        prompt_sets.append(set(df["prompt"].dropna().unique()))

    common = prompt_sets[0]
    for ps in prompt_sets[1:]:
        common = common & ps

    return sorted(common)


def get_always_refused_prompts(
    model_configs: list[ModelConfig],
    max_mean_score: float = 0.1,
    min_prompts: int = 50,
) -> tuple[list[str], bool]:
    """Find prompts that all models always refuse (unattacked).

    Reads the unmodified model outputs (dataset/scored_test_harmful_prompts_cot5_out5.csv)
    and finds prompts where mean StrongREJECT score < ``max_mean_score`` across all
    rollouts, for every model.

    Args:
        model_configs: List of model configs to consider.
        max_mean_score: Threshold below which a prompt counts as "always refused".
        min_prompts: If the universal set is smaller than this, return None to
            signal that per-model sets should be used instead.

    Returns:
        (sorted list of prompt strings, is_universal).
        If the universal set >= min_prompts, returns (universal_set, True).
        Otherwise returns (empty list, False).
    """
    from prompt_attack.models import RESULTS_ROOT

    per_model_refused: dict[str, set[str]] = {}

    for mc in model_configs:
        scored_csv = RESULTS_ROOT / mc.results_subdir / "dataset" / "scored_test_harmful_prompts_cot5_out5.csv"
        if not scored_csv.exists():
            logger.warning(f"Missing baseline CSV: {scored_csv}")
            continue
        df = pd.read_csv(scored_csv, usecols=["prompt", "strongreject_score"])
        per_prompt = df.groupby("prompt")["strongreject_score"].mean()
        refused = set(per_prompt[per_prompt < max_mean_score].index)
        per_model_refused[mc.model_id] = refused
        logger.info(f"{mc.model_id}: {len(refused)} always-refused prompts (mean<{max_mean_score})")

    if not per_model_refused:
        return [], False

    universal = set.intersection(*per_model_refused.values())
    logger.info(f"Universal always-refused: {len(universal)} prompts")

    if len(universal) >= min_prompts:
        return sorted(universal), True
    return [], False


def get_model_refused_prompts(
    model_config: ModelConfig,
    max_mean_score: float = 0.1,
) -> list[str]:
    """Find prompts that a single model always refuses (unattacked)."""
    from prompt_attack.models import RESULTS_ROOT

    scored_csv = RESULTS_ROOT / model_config.results_subdir / "dataset" / "scored_test_harmful_prompts_cot5_out5.csv"
    if not scored_csv.exists():
        logger.warning(f"Missing baseline CSV: {scored_csv}")
        return []
    df = pd.read_csv(scored_csv, usecols=["prompt", "strongreject_score"])
    per_prompt = df.groupby("prompt")["strongreject_score"].mean()
    refused = per_prompt[per_prompt < max_mean_score].index
    logger.info(f"{model_config.model_id}: {len(refused)} always-refused prompts (mean<{max_mean_score})")
    return sorted(refused)


def write_result_to_csv(result: dict, output_csv: str):
    """Append a single result row to CSV with file locking."""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    file_exists = os.path.exists(output_csv)
    row_df = pd.DataFrame([result])

    with open(output_csv, "a" if file_exists else "w", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            row_df.to_csv(f, index=False, header=not file_exists)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def get_output_csv_path(results_dir: str, model_alias: str, beta: float, mode: str,
                         run_tag: str = "") -> str:
    beta_str = f"{beta:.1f}".replace(".", "p")
    tag = f"_{run_tag}" if run_tag else ""
    return os.path.join(results_dir, f"gcg_iris_{model_alias}_beta_{beta_str}_{mode}{tag}.csv")


def load_completed_prompts(output_csv: str) -> set:
    """Load already-completed prompt indices for resume."""
    if os.path.exists(output_csv):
        try:
            df = pd.read_csv(output_csv)
            if "prompt_idx" in df.columns:
                return set(df["prompt_idx"].tolist())
        except Exception:
            pass
    return set()


def worker_init(gpu_id: int, model_config: ModelConfig):
    """Initialize worker process: load model on specified GPU."""
    global _worker_model, _worker_tokenizer, _worker_gpu_id

    _worker_gpu_id = gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    _worker_tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)
    _worker_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        torch_dtype=getattr(torch, model_config.dtype),
        device_map="cuda:0",
        trust_remote_code=True,
    )

    mem = torch.cuda.memory_allocated(0) / 1e9
    logger.info(f"GPU {gpu_id}: Model loaded ({mem:.1f} GB)")


def run_single_experiment(
    prompt_idx: int,
    prompt: str,
    target_string: str,
    model_config: ModelConfig,
    config: GCGConfig,
    output_csv: str,
) -> dict:
    """Run GCG+IRIS on a single prompt."""
    global _worker_model, _worker_tokenizer, _worker_gpu_id

    if _worker_model is None or _worker_tokenizer is None:
        raise RuntimeError("Worker model not initialized")

    try:
        torch.cuda.empty_cache()

        # Override wandb run name with prompt snippet
        if config.wandb_config:
            snippet = "_".join(prompt.strip().split()[:5])
            for ch in '/\\:?*|<>"':
                snippet = snippet.replace(ch, "_")
            snippet = snippet[:50]
            config = GCGConfig(
                **{**config.__dict__,
                   "wandb_config": {**config.wandb_config,
                                    "name": f"p{prompt_idx}_{snippet}",
                                    "config": {
                                        "model": model_config.model_id,
                                        "model_alias": model_config.model_name,
                                        "beta": config.beta,
                                        "refusal_mode": config.refusal_mode,
                                        "refusal_layer": config.refusal_layer,
                                        "prompt_idx": prompt_idx,
                                    }}},
            )

        # Determine refusal vector path
        refusal_path = None
        if config.use_iris and config.beta > 0.0:
            layer = config.refusal_layer or model_config.best_cot_layer
            refusal_path = str(model_config.refusal_dir_path(config.refusal_mode, layer))

        messages = [{"role": "user", "content": prompt}]

        logger.info(f"GPU {_worker_gpu_id}: Starting prompt {prompt_idx} (beta={config.beta})")
        start = time.time()

        result: GCGResult = run(
            _worker_model, _worker_tokenizer, messages,
            target_string, config, refusal_vector_path=refusal_path,
        )

        runtime = time.time() - start

        row = {
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "target_string": target_string,
            "beta": config.beta,
            "model": model_config.model_id,
            "refusal_mode": config.refusal_mode,
            "refusal_layer": config.refusal_layer,
            "gpu_id": _worker_gpu_id,
            "best_response": result.best_answer or "",
            "best_suffix": result.best_string,
            "best_loss": float(result.best_loss),
            "final_loss": float(result.losses[-1]) if result.losses else float("inf"),
            "initial_loss": float(result.losses[0]) if result.losses else float("inf"),
            "runtime_seconds": runtime,
            "num_steps": config.num_steps,
            "success": True,
            "error": None,
            "timestamp": datetime.datetime.now().isoformat(),
            "loss_trajectory": ",".join(f"{l:.4f}" for l in result.losses),
        }

        write_result_to_csv(row, output_csv)
        logger.info(f"GPU {_worker_gpu_id}: Completed prompt {prompt_idx} in {runtime:.1f}s")
        return row

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        gc.collect()
        error_row = _make_error_row(prompt_idx, prompt, target_string, model_config, config, str(e))
        write_result_to_csv(error_row, output_csv)
        return error_row

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        error_row = _make_error_row(prompt_idx, prompt, target_string, model_config, config, error_msg)
        write_result_to_csv(error_row, output_csv)
        return error_row


def _make_error_row(prompt_idx, prompt, target_string, model_config, config, error_msg):
    return {
        "prompt_idx": prompt_idx,
        "prompt": prompt,
        "target_string": target_string,
        "beta": config.beta,
        "model": model_config.model_id,
        "refusal_mode": config.refusal_mode,
        "refusal_layer": config.refusal_layer,
        "gpu_id": _worker_gpu_id,
        "best_response": "",
        "best_suffix": "",
        "best_loss": float("inf"),
        "final_loss": float("inf"),
        "initial_loss": float("inf"),
        "runtime_seconds": 0,
        "num_steps": config.num_steps,
        "success": False,
        "error": error_msg,
        "timestamp": datetime.datetime.now().isoformat(),
        "loss_trajectory": "",
    }


def run_experiments_on_gpu(
    gpu_id: int,
    experiments: list[tuple],
    model_config: ModelConfig,
    config: GCGConfig,
    output_csv: str,
) -> list[dict]:
    """Run a batch of experiments on one GPU."""
    worker_init(gpu_id, model_config)
    results = []
    for prompt_idx, prompt, target_string in experiments:
        result = run_single_experiment(
            prompt_idx, prompt, target_string, model_config, config, output_csv
        )
        results.append(result)
    return results


def run_experiment_batch(
    model_config: ModelConfig,
    config: GCGConfig,
    input_df: pd.DataFrame,
    output_csv: str,
    num_gpus: int = 4,
):
    """Distribute experiments across GPUs with multiprocessing."""
    import multiprocessing as mp

    completed = load_completed_prompts(output_csv)

    experiments = []
    for idx, row in input_df.iterrows():
        if idx in completed:
            logger.info(f"Skipping prompt {idx} (already completed)")
            continue
        experiments.append((idx, row["prompt"], row["target_string"]))

    if not experiments:
        logger.info("All experiments already completed.")
        return

    # Distribute across GPUs
    per_gpu = [[] for _ in range(num_gpus)]
    for i, exp in enumerate(experiments):
        per_gpu[i % num_gpus].append(exp)

    logger.info(f"Running {len(experiments)} experiments across {num_gpus} GPUs")
    for g in range(num_gpus):
        logger.info(f"  GPU {g}: {len(per_gpu[g])} experiments")

    start = time.time()

    with mp.Pool(processes=num_gpus) as pool:
        pairs = [
            (g, per_gpu[g], model_config, config, output_csv)
            for g in range(num_gpus) if per_gpu[g]
        ]
        all_results = pool.starmap(run_experiments_on_gpu, pairs)

    results = [r for gpu_results in all_results for r in gpu_results]
    elapsed = time.time() - start

    success = sum(1 for r in results if r.get("success"))
    failed = len(results) - success
    logger.info(f"Done: {success} succeeded, {failed} failed, {elapsed:.0f}s total")

    # Sort final CSV
    if os.path.exists(output_csv):
        try:
            final = pd.read_csv(output_csv).sort_values("prompt_idx")
            final.to_csv(output_csv, index=False)
        except Exception:
            pass
