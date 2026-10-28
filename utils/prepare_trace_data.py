"""Preprocess scored CSVs into a JSON file for the trace explorer web app.

Finds prompts where all 4 models strongly refuse (base) but strongly comply (ortho),
then exports representative CoT traces and responses for side-by-side comparison.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

MODELS = {
    "DeepSeek-R1-Distill-Llama-8B": {
        "path": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "best_cot_layer": 23,
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "best_cot_layer": 21,
    },
    "Qwen3-8B": {
        "path": "Qwen/Qwen3-8B",
        "best_cot_layer": 23,
    },
    "gpt-oss-20b": {
        "path": "openai/gpt-oss-20b",
        "best_cot_layer": 19,
    },
}


def load_model_data(model_name: str, model_info: dict) -> dict:
    """Load base and ortho scored CSVs for a model, return per-prompt stats."""
    model_dir = RESULTS_DIR / model_info["path"]
    layer = model_info["best_cot_layer"]

    base_path = model_dir / "dataset" / "scored_test_harmful_prompts_cot5_out5.csv"
    # Prefer full scored file, fall back to subset_5
    ortho_path = (
        model_dir
        / "attack_results"
        / f"scored_ortho_output_test_harmful_prompts_cot_layer_{layer}.csv"
    )
    if not ortho_path.exists():
        ortho_path = (
            model_dir
            / "attack_results"
            / f"scored_ortho_output_subset_5_test_harmful_prompts_cot_layer_{layer}.csv"
        )

    print(f"  Loading base: {base_path.name}")
    base_df = pd.read_csv(base_path)
    print(f"    {len(base_df)} rows, {base_df['prompt'].nunique()} unique prompts")

    print(f"  Loading ortho: {ortho_path.name}")
    ortho_df = pd.read_csv(ortho_path)
    print(f"    {len(ortho_df)} rows, {ortho_df['prompt'].nunique()} unique prompts")

    return {"base": base_df, "ortho": ortho_df}


def compute_prompt_means(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean strongreject_score per prompt."""
    return df.groupby("prompt")["strongreject_score"].mean().reset_index()


def pick_representative(df: pd.DataFrame, prompt: str, target_score: float) -> dict:
    """Pick the row closest to the mean score for a prompt."""
    rows = df[df["prompt"] == prompt]
    if rows.empty:
        return None
    # Pick row with score closest to target (mean)
    idx = (rows["strongreject_score"] - target_score).abs().idxmin()
    row = rows.loc[idx]
    return {
        "cot": str(row.get("cot", "")),
        "output": str(row.get("output", "")),
        "score": float(row["strongreject_score"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare trace comparison data")
    parser.add_argument(
        "--refuse-threshold",
        type=float,
        default=0.1,
        help="Max mean score for 'strongly refuses' (default: 0.1)",
    )
    parser.add_argument(
        "--comply-threshold",
        type=float,
        default=0.7,
        help="Min mean score for 'strongly complies' (default: 0.7)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "trace_data.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific models (e.g. --models gpt-oss-20b). Default: all 4.",
    )
    args = parser.parse_args()

    # Filter models if specified
    if args.models:
        selected = {}
        for m in args.models:
            matches = [k for k in MODELS if m.lower() in k.lower()]
            if not matches:
                print(f"Warning: no model matching '{m}'. Available: {list(MODELS.keys())}")
            for k in matches:
                selected[k] = MODELS[k]
        if not selected:
            return
        active_models = selected
    else:
        active_models = MODELS

    print(
        f"Thresholds: refuse ≤ {args.refuse_threshold}, comply ≥ {args.comply_threshold}\n"
    )

    # Load all model data
    all_data = {}
    all_means = {}
    for model_name, model_info in active_models.items():
        print(f"Loading {model_name}...")
        data = load_model_data(model_name, model_info)
        all_data[model_name] = data
        all_means[model_name] = {
            "base": compute_prompt_means(data["base"]),
            "ortho": compute_prompt_means(data["ortho"]),
        }
        print()

    # Find prompts common across selected models (both base and ortho)
    prompt_sets = []
    for model_name in active_models:
        prompt_sets.append(set(all_means[model_name]["base"]["prompt"]))
        prompt_sets.append(set(all_means[model_name]["ortho"]["prompt"]))
    common_prompts = set.intersection(*prompt_sets)
    print(f"Prompts common across selected models: {len(common_prompts)}")

    # Filter by thresholds
    qualifying = []
    for prompt in common_prompts:
        passes = True
        prompt_scores = {}
        for model_name in active_models:
            base_mean = all_means[model_name]["base"]
            base_score = base_mean.loc[
                base_mean["prompt"] == prompt, "strongreject_score"
            ].values[0]

            ortho_mean = all_means[model_name]["ortho"]
            ortho_score = ortho_mean.loc[
                ortho_mean["prompt"] == prompt, "strongreject_score"
            ].values[0]

            if base_score > args.refuse_threshold or ortho_score < args.comply_threshold:
                passes = False
                break

            prompt_scores[model_name] = {
                "base_mean": float(base_score),
                "ortho_mean": float(ortho_score),
            }

        if passes:
            qualifying.append((prompt, prompt_scores))

    print(f"Qualifying prompts ({len(active_models)} model(s) flip): {len(qualifying)}")

    if not qualifying:
        print("\nNo prompts found! Try relaxing thresholds.")
        print("Suggestion: --refuse-threshold 0.2 --comply-threshold 0.5")
        return

    # Build output JSON
    output_prompts = []
    for prompt, scores in sorted(qualifying, key=lambda x: x[0]):
        entry = {"prompt": prompt, "models": {}}
        for model_name in active_models:
            base_rep = pick_representative(
                all_data[model_name]["base"],
                prompt,
                scores[model_name]["base_mean"],
            )
            ortho_rep = pick_representative(
                all_data[model_name]["ortho"],
                prompt,
                scores[model_name]["ortho_mean"],
            )
            if base_rep and ortho_rep:
                base_rep["mean_score"] = scores[model_name]["base_mean"]
                ortho_rep["mean_score"] = scores[model_name]["ortho_mean"]
                entry["models"][model_name] = {
                    "base": base_rep,
                    "ortho": ortho_rep,
                }
        output_prompts.append(entry)

    result = {
        "thresholds": {
            "refuse": args.refuse_threshold,
            "comply": args.comply_threshold,
        },
        "model_names": list(active_models.keys()),
        "prompts": output_prompts,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(output_prompts)} prompts to {args.output}")


if __name__ == "__main__":
    main()
