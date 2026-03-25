"""Generate multiple rollouts from attacked prompts (prompt + GCG suffix).

Uses vLLM for fast batched inference. Loads model once per model alias,
generates N rollouts for each (prompt, suffix) pair across all beta values.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m prompt_attack.generate_rollouts \
        --model deepseek-llama-8b \
        --num-rollouts 5 \
        --refusal-mode cot

    # All models:
    bash bash_scripts/run_scoring.sh
"""

import argparse
import gc
import logging
import os
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from prompt_attack.models import MODEL_REGISTRY, RESULTS_ROOT, get_model_config

logger = logging.getLogger("gcg-iris-score")

BETAS = [0.0, 0.3, 0.5, 0.7, 1.0]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate rollouts from attacked prompts")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--num-rollouts", type=int, default=5)
    parser.add_argument("--refusal-mode", type=str, default="cot",
                        choices=["cot", "baseline"])
    parser.add_argument("--betas", type=str, default=None,
                        help="Comma-separated beta values (default: all)")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--run-tag", type=str, default="",
                        help="Tag appended to filenames to separate runs (e.g. 'v2')")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def extract_cot_and_output(response: str, is_harmony: bool) -> tuple[str, str, bool]:
    """Extract CoT and output portions from a model response."""
    if is_harmony:
        analysis_pattern = r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>'
        final_pattern = r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|$)'
        analysis_match = re.search(analysis_pattern, response, re.DOTALL)
        final_match = re.search(final_pattern, response, re.DOTALL)
        if not analysis_match:
            return response, "", False
        cot_part = analysis_match.group(1)
        output_part = final_match.group(1) if final_match else ""
        return cot_part, output_part, True

    think_pattern = r'(.*?</think>)(.*)'
    match = re.search(think_pattern, response, re.DOTALL)
    if match:
        return match.group(1), match.group(2), True
    return response, "", False


def get_attack_csv_path(model_alias: str, beta: float, mode: str, run_tag: str = "") -> Path:
    """Get path to attack result CSV."""
    mc = get_model_config(model_alias)
    beta_str = f"{beta:.1f}".replace(".", "p")
    tag = f"_{run_tag}" if run_tag else ""
    return (
        RESULTS_ROOT / mc.results_subdir / "prompt_attack"
        / f"gcg_iris_{model_alias}_beta_{beta_str}_{mode}{tag}.csv"
    )


def get_rollout_csv_path(model_alias: str, beta: float, mode: str, run_tag: str = "") -> Path:
    """Get output path for rollouts CSV."""
    mc = get_model_config(model_alias)
    beta_str = f"{beta:.1f}".replace(".", "p")
    tag = f"_{run_tag}" if run_tag else ""
    return (
        RESULTS_ROOT / mc.results_subdir / "prompt_attack"
        / f"rollouts_{model_alias}_beta_{beta_str}_{mode}{tag}.csv"
    )


def load_attack_results(model_alias: str, beta: float, mode: str, run_tag: str = "") -> pd.DataFrame:
    """Load attack results and return successful rows with prompt + suffix."""
    csv_path = get_attack_csv_path(model_alias, beta, mode, run_tag)
    if not csv_path.exists():
        logger.warning(f"Attack CSV not found: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    # Filter to successful attacks only
    df = df[df["success"] == True].reset_index(drop=True)
    logger.info(f"Loaded {len(df)} successful attacks from {csv_path.name}")
    return df[["prompt_idx", "prompt", "best_suffix", "target_string", "best_loss"]]


def generate_rollouts(
    llm: LLM,
    tokenizer,
    attack_df: pd.DataFrame,
    model_config,
    num_rollouts: int,
    sampling_params: SamplingParams,
) -> list[dict]:
    """Generate N rollouts per attacked prompt."""
    # Build attacked prompts: prompt + " " + suffix
    attacked_prompts = []
    prompt_indices = []
    rollout_indices = []

    for _, row in attack_df.iterrows():
        attacked = f"{row['prompt']} {row['best_suffix']}"
        for r in range(num_rollouts):
            attacked_prompts.append(attacked)
            prompt_indices.append(row["prompt_idx"])
            rollout_indices.append(r + 1)

    # Apply chat template
    formatted = []
    for prompt in attacked_prompts:
        chat = [{"role": "user", "content": prompt}]
        formatted.append(tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        ))

    logger.info(f"Generating {len(formatted)} responses ({len(attack_df)} prompts x {num_rollouts} rollouts)")
    outputs = llm.generate(formatted, sampling_params)

    results = []
    is_harmony = model_config.is_harmony
    for i, output in enumerate(outputs):
        full_response = tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=False)
        cot_part, output_part, has_cot = extract_cot_and_output(full_response, is_harmony)

        row_idx = prompt_indices[i]
        attack_row = attack_df[attack_df["prompt_idx"] == row_idx].iloc[0]

        results.append({
            "prompt_idx": row_idx,
            "prompt": attack_row["prompt"],
            "suffix": attack_row["best_suffix"],
            "target_string": attack_row["target_string"],
            "best_loss": attack_row["best_loss"],
            "rollout_idx": rollout_indices[i],
            "full_response": full_response,
            "cot": cot_part,
            "output": output_part,
            "has_cot": has_cot,
        })

    return results


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(name)s] %(message)s")

    model_config = get_model_config(args.model)
    betas = [float(b) for b in args.betas.split(",")] if args.betas else BETAS

    # Check which betas still need rollouts
    betas_to_run = []
    for beta in betas:
        out_path = get_rollout_csv_path(args.model, beta, args.refusal_mode, args.run_tag)
        if out_path.exists():
            existing = pd.read_csv(out_path)
            if len(existing) > 0:
                logger.info(f"Rollouts already exist for beta={beta} ({len(existing)} rows), skipping")
                continue
        attack_csv = get_attack_csv_path(args.model, beta, args.refusal_mode, args.run_tag)
        if not attack_csv.exists():
            logger.warning(f"No attack CSV for beta={beta}, skipping")
            continue
        betas_to_run.append(beta)

    if not betas_to_run:
        logger.info("All rollouts already generated.")
        return

    logger.info(f"Model: {model_config.model_id}")
    logger.info(f"Betas to generate: {betas_to_run}")

    # Determine tensor parallel size
    tp_size = args.tensor_parallel_size or torch.cuda.device_count()

    # Load model once via vLLM
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_id, trust_remote_code=True)
    llm = LLM(
        model=model_config.model_id,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    logger.info("Model loaded via vLLM")

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Generate rollouts for each beta
    for beta in betas_to_run:
        logger.info(f"\n--- Beta={beta} ---")
        attack_df = load_attack_results(args.model, beta, args.refusal_mode, args.run_tag)
        if attack_df.empty:
            continue

        results = generate_rollouts(
            llm, tokenizer, attack_df, model_config,
            args.num_rollouts, sampling_params,
        )

        # Save
        out_path = get_rollout_csv_path(args.model, beta, args.refusal_mode, args.run_tag)
        out_df = pd.DataFrame(results)
        out_df["beta"] = beta
        out_df["model"] = model_config.model_id
        out_df["refusal_mode"] = args.refusal_mode
        out_df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(out_df)} rollouts to {out_path}")

    del llm, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Done generating rollouts")


if __name__ == "__main__":
    main()
