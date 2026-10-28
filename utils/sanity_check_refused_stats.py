"""Sanity-check stats on the always-refused subset.

Recomputes mean/std/n for `strongreject_score` on the same prompt subset and
result files used by the poster violin plots in
`utils/plot_boxplot_comparison.py::make_poster_plot_baseline` (and friends).

Run:
    uv run python -m utils.sanity_check_refused_stats
"""

import pandas as pd
import numpy as np


MODEL_NAMES = [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen/Qwen3-8B",
    "openai/gpt-oss-20b",
]
DISPLAY_NAMES = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "DeepSeek-R1-Distill-Qwen-7B",
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "openai/gpt-oss-20b": "GPT-OSS-20B",
}
REFUSAL_THRESHOLD = 0.4
N_REPS = 5 * 5


def load_before_dfs():
    return {
        m: pd.read_csv(f"results/{m}/dataset/scored_test_harmful_prompts_cot5_out5.csv")
        for m in MODEL_NAMES
    }


def get_common_refused_prompts(before_dfs):
    per_model = {}
    for m, df in before_dfs.items():
        prompt_max = df.groupby("prompt")["strongreject_score"].max()
        per_model[m] = set(prompt_max[prompt_max < REFUSAL_THRESHOLD].index)
    common = set.intersection(*per_model.values())
    print(f"Common always-refused prompts (max score < {REFUSAL_THRESHOLD}): "
          f"{len(common)}")
    for m in MODEL_NAMES:
        print(f"  {DISPLAY_NAMES[m]:32s} refused-only: {len(per_model[m])}")
    return common


def stats(scores: pd.Series) -> dict:
    s = pd.Series(scores).dropna().clip(0, 1)
    return {
        "n": int(len(s)),
        "mean": float(s.mean()) if len(s) else float("nan"),
        "std": float(s.std(ddof=0)) if len(s) else float("nan"),
        "std_ddof1": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
        "median": float(s.median()) if len(s) else float("nan"),
    }


def fmt(s: dict) -> str:
    return (f"n={s['n']:5d}  mean={s['mean']:.4f}  "
            f"std(pop)={s['std']:.4f}  std(samp)={s['std_ddof1']:.4f}  "
            f"median={s['median']:.4f}")


def zero_pad(scores: pd.Series, expected: int) -> pd.Series:
    s = pd.Series(scores).dropna().clip(0, 1).reset_index(drop=True)
    n_missing = expected - len(s)
    if n_missing > 0:
        s = pd.concat([s, pd.Series(np.zeros(n_missing))], ignore_index=True)
    return s


def collect_zero_padded_stats(before_dfs, common):
    """Return {model: {'cot': (mean,std), 'eop': (mean,std)}} with zero-padding."""
    expected = len(common) * N_REPS
    out = {}
    for m in MODEL_NAMES:
        with open(f"results/{m}/.best_layer_cot") as f:
            cot_layer = int(f.read().strip())
        with open(f"results/{m}/.best_layer_baseline") as f:
            bl_layer = int(f.read().strip())
        cot_df = pd.read_csv(
            f"results/{m}/attack_results/"
            f"scored_ortho_output_test_harmful_prompts_cot_layer_{cot_layer}.csv"
        )
        bl_df = pd.read_csv(
            f"results/{m}/attack_results/"
            f"scored_ortho_output_test_harmful_prompts_baseline_layer_{bl_layer}.csv"
        )
        cot_padded = zero_pad(cot_df[cot_df["prompt"].isin(common)]["strongreject_score"], expected)
        bl_padded = zero_pad(bl_df[bl_df["prompt"].isin(common)]["strongreject_score"], expected)
        out[m] = {
            "cot": (cot_padded.mean(), cot_padded.std(ddof=1)),
            "eop": (bl_padded.mean(), bl_padded.std(ddof=1)),
            "cot_layer": cot_layer,
            "bl_layer": bl_layer,
        }
    return out


def print_latex_table(stats_by_model):
    rows = []
    for m in MODEL_NAMES:
        s = stats_by_model[m]
        cot_m, cot_s = s["cot"]
        eop_m, eop_s = s["eop"]
        rows.append(
            f"  {DISPLAY_NAMES[m]:<32s} & "
            f"${cot_m:.3f} \\pm {cot_s:.3f}$ & "
            f"${eop_m:.3f} \\pm {eop_s:.3f}$ \\\\"
        )
    print("\\begin{table}[ht]")
    print("  \\centering")
    print("  \\caption{Mean ($\\pm$ std) StrongREJECT scores after intervention on "
          "common-refused prompt subset (StrongREJECT score $< 0.4$). Missing "
          "generations are zero-padded to the expected count.}")
    print("  \\label{tab:cot_vs_eop}")
    print("  \\begin{tabular}{lcc}")
    print("  \\toprule")
    print("  Model & CoT & End-of-prompt \\\\")
    print("  \\midrule")
    for r in rows:
        print(r)
    print("  \\bottomrule")
    print("  \\end{tabular}")
    print("\\end{table}")


def main():
    before_dfs = load_before_dfs()
    common = get_common_refused_prompts(before_dfs)
    expected = len(common) * N_REPS
    print(f"Expected per-model rows on refused subset: "
          f"{len(common)} prompts * {N_REPS} reps = {expected}")
    print()

    for m in MODEL_NAMES:
        with open(f"results/{m}/.best_layer_cot") as f:
            cot_layer = int(f.read().strip())
        with open(f"results/{m}/.best_layer_baseline") as f:
            bl_layer = int(f.read().strip())

        cot_df = pd.read_csv(
            f"results/{m}/attack_results/"
            f"scored_ortho_output_test_harmful_prompts_cot_layer_{cot_layer}.csv"
        )
        bl_df = pd.read_csv(
            f"results/{m}/attack_results/"
            f"scored_ortho_output_test_harmful_prompts_baseline_layer_{bl_layer}.csv"
        )

        before_df = before_dfs[m]
        before_filt = before_df[before_df["prompt"].isin(common)]
        cot_filt = cot_df[cot_df["prompt"].isin(common)]
        bl_filt = bl_df[bl_df["prompt"].isin(common)]

        print("=" * 88)
        print(f"{DISPLAY_NAMES[m]}  "
              f"(CoT layer {cot_layer}, end-of-prompt layer {bl_layer})")
        print("-" * 88)
        print(f"  unique prompts in refused subset present in file:  "
              f"before={before_filt['prompt'].nunique()}, "
              f"cot={cot_filt['prompt'].nunique()}, "
              f"prompt={bl_filt['prompt'].nunique()}")

        for label, scores in [
            ("Before               ", before_filt["strongreject_score"]),
            ("After CoT            ", cot_filt["strongreject_score"]),
            ("After end-of-prompt  ", bl_filt["strongreject_score"]),
        ]:
            print(f"  {label} {fmt(stats(scores))}")

        print("  -- with zero-padding to expected n ({}):".format(expected))
        for label, scores in [
            ("Before               ", before_filt["strongreject_score"]),
            ("After CoT            ", cot_filt["strongreject_score"]),
            ("After end-of-prompt  ", bl_filt["strongreject_score"]),
        ]:
            padded = zero_pad(scores, expected)
            print(f"  {label} {fmt(stats(padded))}")

        print()


if __name__ == "__main__":
    main()
    print()
    print("=" * 88)
    print("LaTeX table (zero-padded, sample std with ddof=1):")
    print("=" * 88)
    before_dfs = load_before_dfs()
    common = get_common_refused_prompts(before_dfs)
    print()
    print_latex_table(collect_zero_padded_stats(before_dfs, common))
