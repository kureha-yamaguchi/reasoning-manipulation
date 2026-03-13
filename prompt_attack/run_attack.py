"""CLI entrypoint for GCG+IRIS attacks.

Usage:
    python -m prompt_attack.run_attack \
        --model deepseek-llama-8b \
        --beta 0.5 \
        --num-gpus 4 \
        --num-steps 150 \
        --refusal-mode cot \
        --target-tokens 20
"""

import argparse
import logging
import os
import warnings

warnings.filterwarnings("ignore", message="Flash Attention defaults to a non-deterministic algorithm")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer

from prompt_attack.config import GCGConfig
from prompt_attack.models import MODEL_REGISTRY, get_model_config
from prompt_attack.runner import (
    get_common_prompts,
    get_output_csv_path,
    prepare_input_data,
    run_experiment_batch,
)

logger = logging.getLogger("gcg-iris")


def parse_args():
    parser = argparse.ArgumentParser(description="GCG+IRIS attack runner")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model alias from registry")
    parser.add_argument("--beta", type=float, required=True,
                        help="IRIS weight (0.0=pure GCG, 1.0=pure IRIS)")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=150)
    parser.add_argument("--refusal-mode", type=str, default="cot",
                        choices=["cot", "baseline"])
    parser.add_argument("--target-tokens", type=int, default=20,
                        help="Number of CoT tokens to use as target")
    parser.add_argument("--extended-gen-tokens", type=int, default=15)
    parser.add_argument("--eval-frequency", type=int, default=25)
    parser.add_argument("--num-prompts", type=int, default=50,
                        help="Number of prompts to attack")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Results dir (default: per-model results/<org>/<model>/prompt_attack/)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-common-prompts", action="store_true",
                        help="Use intersection of prompts across all models")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(name)s] %(message)s")

    model_config = get_model_config(args.model)
    logger.info(f"Model: {model_config.model_id}")
    logger.info(f"Beta: {args.beta}, Mode: {args.refusal_mode}, Steps: {args.num_steps}")

    # Determine refusal layer
    refusal_layer = (
        model_config.best_cot_layer if args.refusal_mode == "cot"
        else model_config.best_baseline_layer
    )

    config = GCGConfig(
        num_steps=args.num_steps,
        beta=args.beta,
        use_iris=args.beta > 0.0,
        refusal_mode=args.refusal_mode,
        refusal_layer=refusal_layer,
        extended_gen_tokens=args.extended_gen_tokens,
        target_tokens=args.target_tokens,
        eval_frequency=args.eval_frequency,
        seed=args.seed,
        verbosity="DEBUG" if args.verbose else "INFO",
        think_start_token=model_config.think_start,
        think_end_token=model_config.think_end,
        wandb_config={
            "entity": "reasoning_attacks",
            "project": "gcg-iris-clean",
            "name": f"{args.model}_beta{args.beta}_{args.refusal_mode}",
        },
    )

    # Load tokenizer for data prep
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)

    # Get prompt subset
    prompt_subset = None
    if args.use_common_prompts:
        all_configs = [get_model_config(alias) for alias in MODEL_REGISTRY]
        common = get_common_prompts(all_configs, args.refusal_mode)
        logger.info(f"Found {len(common)} common prompts across all models")
        prompt_subset = common[:args.num_prompts]
    else:
        # Just load and take first N unique prompts
        import pandas as pd
        csv_path = model_config.input_csv_path(args.refusal_mode)
        df = pd.read_csv(csv_path)
        all_prompts = sorted(df["prompt"].dropna().unique())
        prompt_subset = all_prompts[:args.num_prompts]

    # Prepare data
    input_df = prepare_input_data(
        model_config, tokenizer, args.target_tokens, args.refusal_mode, prompt_subset
    )

    # Default results dir: results/<org>/<model>/prompt_attack/
    results_dir = args.results_dir
    if results_dir is None:
        from prompt_attack.models import RESULTS_ROOT
        results_dir = str(RESULTS_ROOT / model_config.results_subdir / "prompt_attack")

    output_csv = get_output_csv_path(
        results_dir, args.model, args.beta, args.refusal_mode
    )
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"Will attack {len(input_df)} prompts, output: {output_csv}")

    if args.dry_run:
        logger.info("DRY RUN — not executing")
        for i, row in input_df.head(5).iterrows():
            logger.info(f"  [{i}] {row['prompt'][:60]}... -> target: {row['target_string'][:40]}...")
        if len(input_df) > 5:
            logger.info(f"  ... and {len(input_df) - 5} more")
        return

    run_experiment_batch(model_config, config, input_df, output_csv, args.num_gpus)


if __name__ == "__main__":
    main()
