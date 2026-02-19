"""
Plot line graph of generation success percentage per layer for all models.
One subplot per model, with lines for each intervention variant.
"""

import json
import matplotlib.pyplot as plt
import os

MAX_ROWS = 125  # 5 prompts * 5 cot_reps * 5 output_reps

MODELS = [
    {
        "name": "DeepSeek-R1-Distill-Llama-8B",
        "results_dir": "results/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/attack_results",
        "layers": [11, 13, 15, 17, 19, 21, 23],
        "layer_suffix": "11_13_15_17_19_21_23",
    },
    {
        "name": "DeepSeek-R1-Distill-Qwen-7B",
        "results_dir": "results/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/attack_results",
        "layers": [11, 13, 15, 17, 19, 21, 23],
        "layer_suffix": "11_13_15_17_19_21_23",
    },
    {
        "name": "Qwen3-8B",
        "results_dir": "results/Qwen/Qwen3-8B/attack_results",
        "layers": [11, 13, 15, 17, 19, 21, 23],
        "layer_suffix": "11_13_15_17_19_21_23",
    },
    {
        "name": "GPT-OSS-20B",
        "results_dir": "results/openai/gpt-oss-20b/attack_results",
        "layers": [7, 9, 11, 13, 15, 17, 19],
        "layer_suffix": "7_9_11_13_15_17_19",
    },
]

VARIANT_FILES = [
    ("CoT (harmful)", "layer_statistics_cot_layers_{suffix}.json"),
    ("Baseline (harmful)", "layer_statistics_baseline_layers_{suffix}.json"),
    ("CoT (harmless)", "layer_statistics_cot_layers_{suffix}_harmless.json"),
    ("Baseline (harmless)", "layer_statistics_baseline_layers_{suffix}_harmless.json"),
]

COLORS = {
    "CoT (harmful)": "#1f77b4",
    "Baseline (harmful)": "#ff7f0e",
    "CoT (harmless)": "#2ca02c",
    "Baseline (harmless)": "#d62728",
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, model in enumerate(MODELS):
    ax = axes[idx]
    layers = model["layers"]
    suffix = model["layer_suffix"]

    for variant_name, file_template in VARIANT_FILES:
        filename = file_template.format(suffix=suffix)
        filepath = os.path.join(model["results_dir"], filename)

        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping")
            continue

        with open(filepath) as f:
            data = json.load(f)

        percentages = []
        for layer in layers:
            count = data[str(layer)]["count"]
            percentages.append(count / MAX_ROWS * 100)

        ax.plot(layers, percentages, marker="o", label=variant_name,
                color=COLORS[variant_name])

    ax.set_xlabel("Layer")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title(model["name"])
    ax.set_xticks(layers)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("Generation Success Rate by Layer", fontsize=14, fontweight="bold")
plt.tight_layout()

out_path = "figures/success_rate_all_models.png"
os.makedirs("figures", exist_ok=True)
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
plt.close()
