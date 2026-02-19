"""
Plot line graph of generation success percentage per layer for GPT-OSS ortho variants.
"""

import json
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results/openai/gpt-oss-20b/attack_results"
MAX_ROWS = 125  # 5 prompts * 5 cot_reps * 5 output_reps

VARIANTS = {
    "CoT (harmful)": "layer_statistics_cot_layers_7_9_11_13_15_17_19.json",
    "Baseline (harmful)": "layer_statistics_baseline_layers_7_9_11_13_15_17_19.json",
    "CoT (harmless)": "layer_statistics_cot_layers_7_9_11_13_15_17_19_harmless.json",
    "Baseline (harmless)": "layer_statistics_baseline_layers_7_9_11_13_15_17_19_harmless.json",
}

LAYERS = [7, 9, 11, 13, 15, 17, 19]

fig, ax = plt.subplots(figsize=(8, 5))

for variant_name, filename in VARIANTS.items():
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath) as f:
        data = json.load(f)

    percentages = []
    for layer in LAYERS:
        count = data[str(layer)]["count"]
        percentages.append(count / MAX_ROWS * 100)

    ax.plot(LAYERS, percentages, marker="o", label=variant_name)

ax.set_xlabel("Layer")
ax.set_ylabel("Success Rate (%)")
ax.set_title("GPT-OSS-20B: Generation Success Rate by Layer")
ax.set_xticks(LAYERS)
ax.set_ylim(-5, 105)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "figures/gpt_oss_success_rate_by_layer.png"
os.makedirs("figures", exist_ok=True)
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
plt.close()
