"""Model configuration registry for GCG+IRIS attacks."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    org: str
    model_name: str
    num_layers: int
    dtype: str  # "float16" or "bfloat16"
    think_start: str
    think_end: str
    best_cot_layer: int
    best_baseline_layer: int
    results_subdir: str  # e.g. "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    is_moe: bool = False
    is_harmony: bool = False  # gpt-oss harmony format

    def refusal_dir_path(self, mode: str, layer: Optional[int] = None) -> Path:
        if layer is None:
            layer = self.best_cot_layer if mode == "cot" else self.best_baseline_layer
        return RESULTS_ROOT / self.results_subdir / "refusal_dir" / f"refusal_dir_{mode}_layer_{layer}.pt"

    def input_csv_path(self, mode: str, layer: Optional[int] = None) -> Path:
        if layer is None:
            layer = self.best_cot_layer if mode == "cot" else self.best_baseline_layer
        return (
            RESULTS_ROOT / self.results_subdir / "attack_results"
            / f"scored_ortho_output_test_harmful_prompts_{mode}_layer_{layer}.csv"
        )


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "deepseek-llama-8b": ModelConfig(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        org="deepseek-ai",
        model_name="DeepSeek-R1-Distill-Llama-8B",
        num_layers=32,
        dtype="float16",
        think_start="<think>",
        think_end="</think>",
        best_cot_layer=23,
        best_baseline_layer=15,
        results_subdir="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    ),
    "deepseek-qwen-7b": ModelConfig(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        org="deepseek-ai",
        model_name="DeepSeek-R1-Distill-Qwen-7B",
        num_layers=28,
        dtype="float16",
        think_start="<think>",
        think_end="</think>",
        best_cot_layer=21,
        best_baseline_layer=17,
        results_subdir="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ),
    "qwen3-8b": ModelConfig(
        model_id="Qwen/Qwen3-8B",
        org="Qwen",
        model_name="Qwen3-8B",
        num_layers=36,
        dtype="float16",
        think_start="<think>",
        think_end="</think>",
        best_cot_layer=23,
        best_baseline_layer=17,
        results_subdir="Qwen/Qwen3-8B",
    ),
    "gpt-oss-20b": ModelConfig(
        model_id="openai/gpt-oss-20b",
        org="openai",
        model_name="gpt-oss-20b",
        num_layers=24,
        dtype="bfloat16",
        think_start="<|channel|>analysis<|message|>",
        think_end="<|end|>",
        best_cot_layer=19,
        best_baseline_layer=15,
        results_subdir="openai/gpt-oss-20b",
        is_moe=True,
        is_harmony=True,
    ),
}


def get_model_config(alias_or_id: str) -> ModelConfig:
    if alias_or_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[alias_or_id]
    for cfg in MODEL_REGISTRY.values():
        if cfg.model_id == alias_or_id:
            return cfg
    raise KeyError(f"Unknown model: {alias_or_id}. Available: {list(MODEL_REGISTRY.keys())}")
