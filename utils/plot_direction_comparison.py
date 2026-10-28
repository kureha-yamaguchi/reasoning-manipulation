"""
Compare CoT and end-of-prompt refusal directions: cosine similarity and dot product
across all models and layers. Produces a heatmap figure.

Usage:
    uv run python -m utils.plot_direction_comparison
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def main():
    models = {
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
            "display": "DeepSeek-R1-Distill-Llama-8B",
            "layers": [13, 15, 17, 19, 21, 23],
        },
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
            "display": "DeepSeek-R1-Distill-Qwen-7B",
            "layers": [13, 15, 17, 19, 21, 23],
        },
        "Qwen/Qwen3-8B": {
            "display": "Qwen3-8B",
            "layers": [13, 15, 17, 19, 21, 23],
        },
        "openai/gpt-oss-20b": {
            "display": "GPT-OSS-20B",
            "layers": [7, 9, 11, 13, 15, 17, 19],
        },
    }

    # Collect results
    results = {}
    for model_path, info in models.items():
        display = info["display"]
        results[display] = {"layers": [], "cosine_sim": [], "dot_product": []}

        for layer in info["layers"]:
            cot_path = f"results/{model_path}/refusal_dir/refusal_dir_cot_layer_{layer}.pt"
            bl_path = f"results/{model_path}/refusal_dir/refusal_dir_baseline_layer_{layer}.pt"

            cot_dir = torch.load(cot_path, map_location="cpu", weights_only=True).float()
            bl_dir = torch.load(bl_path, map_location="cpu", weights_only=True).float()

            # Normalize
            cot_norm = cot_dir / cot_dir.norm()
            bl_norm = bl_dir / bl_dir.norm()

            cosine = (cot_norm @ bl_norm).item()
            dot = (cot_dir @ bl_dir).item()

            results[display]["layers"].append(layer)
            results[display]["cosine_sim"].append(cosine)
            results[display]["dot_product"].append(dot)

            print(f"{display} layer {layer}: cosine={cosine:.4f}, dot={dot:.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    color = "#2176ae"

    for ax, (display, data) in zip(axes.flat, results.items()):
        layers = data["layers"]
        cosines = data["cosine_sim"]

        ax.bar(layers, cosines, width=1.2, color=color, alpha=0.85, zorder=3)

        # Label each bar
        for l, c in zip(layers, cosines):
            ax.text(l, c + 0.02, f"{c:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(display, fontsize=13, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Cosine similarity", fontsize=11)
        ax.set_ylim(0, 0.85)
        ax.set_xticks(layers)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Cosine similarity between CoT and end-of-prompt\nrefusal directions (per layer, per model)",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()

    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    for ext in ["png", "pdf"]:
        plt.savefig(f"{output_dir}/direction_comparison.{ext}", dpi=300, bbox_inches="tight")
    print(f"\nSaved to: {output_dir}/direction_comparison.png/pdf")
    plt.close()


if __name__ == "__main__":
    main()
