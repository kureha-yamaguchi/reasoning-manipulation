"""Score generated rollouts using StrongREJECT evaluator via local vLLM.

Also extracts counterfactual (unattacked) scores for the same prompts from
existing scored CSVs.

Requires STRONGREJECT_VLLM_URL to be set (e.g., http://localhost:8000/v1).

Usage:
    STRONGREJECT_VLLM_URL=http://localhost:8000/v1 \
    uv run python -m prompt_attack.score_rollouts \
        --model deepseek-llama-8b \
        --refusal-mode cot

    # Extract counterfactual only:
    uv run python -m prompt_attack.score_rollouts \
        --model deepseek-llama-8b \
        --counterfactual-only
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from prompt_attack.models import MODEL_REGISTRY, RESULTS_ROOT, get_model_config
from utils.compute_score_outputs import compute_scores, load_data_efficiently, save_scored_csv

logger = logging.getLogger("gcg-iris-score")

BETAS = [0.0, 0.3, 0.5, 0.7, 1.0]


def parse_args():
    parser = argparse.ArgumentParser(description="Score attack rollouts with StrongREJECT")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--refusal-mode", type=str, default="cot",
                        choices=["cot", "baseline"])
    parser.add_argument("--betas", type=str, default=None,
                        help="Comma-separated beta values (default: all)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--counterfactual-only", action="store_true",
                        help="Only extract counterfactual scores, skip rollout scoring")
    parser.add_argument("--run-tag", type=str, default="",
                        help="Tag appended to filenames to separate runs (e.g. 'v2')")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def get_rollout_csv_path(model_alias: str, beta: float, mode: str, run_tag: str = "") -> Path:
    mc = get_model_config(model_alias)
    beta_str = f"{beta:.1f}".replace(".", "p")
    tag = f"_{run_tag}" if run_tag else ""
    return (
        RESULTS_ROOT / mc.results_subdir / "prompt_attack"
        / f"rollouts_{model_alias}_beta_{beta_str}_{mode}{tag}.csv"
    )


def get_scored_rollout_csv_path(model_alias: str, beta: float, mode: str, run_tag: str = "") -> Path:
    mc = get_model_config(model_alias)
    beta_str = f"{beta:.1f}".replace(".", "p")
    tag = f"_{run_tag}" if run_tag else ""
    return (
        RESULTS_ROOT / mc.results_subdir / "prompt_attack"
        / f"scored_rollouts_{model_alias}_beta_{beta_str}_{mode}{tag}.csv"
    )


def get_counterfactual_csv_path(model_alias: str, mode: str, run_tag: str = "") -> Path:
    mc = get_model_config(model_alias)
    tag = f"_{run_tag}" if run_tag else ""
    return (
        RESULTS_ROOT / mc.results_subdir / "prompt_attack"
        / f"counterfactual_{model_alias}_{mode}{tag}.csv"
    )


def score_rollouts(model_alias: str, beta: float, mode: str, batch_size: int, run_tag: str = ""):
    """Score rollouts for a single (model, beta) pair."""
    rollout_path = get_rollout_csv_path(model_alias, beta, mode, run_tag)
    scored_path = get_scored_rollout_csv_path(model_alias, beta, mode, run_tag)

    if scored_path.exists():
        logger.info(f"Scored rollouts already exist: {scored_path.name}, skipping")
        return

    if not rollout_path.exists():
        logger.warning(f"Rollouts not found: {rollout_path}, skipping")
        return

    # Load using existing utility
    all_rows, fieldnames = load_data_efficiently(str(rollout_path))
    logger.info(f"Scoring {len(all_rows)} rollouts for beta={beta}")

    # Score using existing utility (returns dict with score, refusal, convincingness, specificity)
    scores = compute_scores(all_rows, batch_size)

    # Save using existing utility
    save_scored_csv(all_rows, scores, fieldnames, str(scored_path))
    logger.info(f"Saved scored rollouts to {scored_path.name}")

    # Print summary
    if "score" in scores:
        mean_score = sum(scores["score"]) / len(scores["score"])
        logger.info(f"  Mean StrongREJECT score: {mean_score:.3f}")


def extract_counterfactual(model_alias: str, mode: str, run_tag: str = ""):
    """Extract counterfactual scores for the attack's 50 prompts from existing scored CSVs."""
    mc = get_model_config(model_alias)
    output_path = get_counterfactual_csv_path(model_alias, mode, run_tag)

    if output_path.exists():
        logger.info(f"Counterfactual already exists: {output_path.name}, skipping")
        return

    # Load the existing scored clean model output
    scored_csv = RESULTS_ROOT / mc.results_subdir / "dataset" / "scored_test_harmful_prompts_cot5_out5.csv"
    if not scored_csv.exists():
        logger.error(f"Scored counterfactual CSV not found: {scored_csv}")
        return

    scored_df = pd.read_csv(scored_csv)
    logger.info(f"Loaded {len(scored_df)} rows from {scored_csv.name}")

    # Get the 50 attack prompts from any beta's attack CSV
    attack_prompts = None
    for beta in BETAS:
        beta_str = f"{beta:.1f}".replace(".", "p")
        attack_csv = (
            RESULTS_ROOT / mc.results_subdir / "prompt_attack"
            / f"gcg_iris_{model_alias}_beta_{beta_str}_{mode}.csv"
        )
        if attack_csv.exists():
            adf = pd.read_csv(attack_csv)
            attack_prompts = set(adf["prompt"].dropna().unique())
            logger.info(f"Found {len(attack_prompts)} attack prompts from {attack_csv.name}")
            break

    if attack_prompts is None:
        logger.error("No attack CSVs found to determine prompt set")
        return

    # Filter scored CSV to attack prompts
    cf_df = scored_df[scored_df["prompt"].isin(attack_prompts)].copy()

    # The pipeline data has 5 cot_reps x 5 output_reps = 25 rows/prompt, but the
    # attack rollouts only have NUM_ROLLOUTS (typically 5) per prompt.  Sample one
    # output_rep per cot_rep so the counterfactual has the same 5 rows/prompt.
    if "cot_rep_n" in cf_df.columns and "output_rep_n" in cf_df.columns:
        before = len(cf_df)
        cf_df = cf_df[cf_df["output_rep_n"] == 1]
        logger.info(f"Sampled output_rep_n==1: {before} -> {len(cf_df)} rows (5 per prompt)")

    cf_df["model"] = mc.model_id
    cf_df["model_alias"] = model_alias

    logger.info(f"Extracted {len(cf_df)} counterfactual rows for {cf_df['prompt'].nunique()} prompts")
    cf_df.to_csv(output_path, index=False)
    logger.info(f"Saved counterfactual to {output_path.name}")

    # Summary stats
    if "strongreject_score" in cf_df.columns:
        mean_score = cf_df["strongreject_score"].mean()
        logger.info(f"  Mean counterfactual StrongREJECT score: {mean_score:.3f}")


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(name)s] %(message)s")

    betas = [float(b) for b in args.betas.split(",")] if args.betas else BETAS

    # Always extract counterfactual
    logger.info(f"=== Extracting counterfactual for {args.model} ===")
    extract_counterfactual(args.model, args.refusal_mode, args.run_tag)

    if args.counterfactual_only:
        return

    # Check STRONGREJECT_VLLM_URL
    vllm_url = os.environ.get("STRONGREJECT_VLLM_URL")
    if not vllm_url:
        logger.error("STRONGREJECT_VLLM_URL not set. Start the vLLM judge server first.")
        logger.error("Example: export STRONGREJECT_VLLM_URL=http://localhost:8000/v1")
        return

    logger.info(f"Using StrongREJECT vLLM at: {vllm_url}")

    # Score rollouts for each beta
    for beta in betas:
        logger.info(f"\n--- Scoring beta={beta} ---")
        score_rollouts(args.model, beta, args.refusal_mode, args.batch_size, args.run_tag)

    logger.info("Done scoring rollouts")


if __name__ == "__main__":
    main()
