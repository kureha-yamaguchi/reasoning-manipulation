"""Activation hook manager for capturing layer activations."""

import logging
from typing import Optional

import torch
import transformers

logger = logging.getLogger("gcg-iris")


class ActivationHookManager:
    """Registers forward hooks on model layers to capture activations."""

    def __init__(self, model: transformers.PreTrainedModel, layer_indices: list[int]):
        self.model = model
        self.layer_indices = layer_indices
        self.activations: dict[int, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._register()

    def _register(self):
        if not (hasattr(self.model, "model") and hasattr(self.model.model, "layers")):
            raise ValueError("Model does not have model.model.layers — unsupported architecture")

        total_layers = len(self.model.model.layers)

        for idx in self.layer_indices:
            if idx < 0 or idx >= total_layers:
                logger.warning(f"Layer {idx} out of range (model has {total_layers} layers), skipping")
                continue
            layer = self.model.model.layers[idx]
            if not hasattr(layer, "input_layernorm"):
                logger.warning(f"Layer {idx} has no input_layernorm, skipping")
                continue

            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    act = input[0] if isinstance(input, tuple) else input
                    self.activations[layer_idx] = act
                return hook_fn

            h = layer.input_layernorm.register_forward_hook(make_hook(idx))
            self._hooks.append(h)

        logger.info(f"Registered activation hooks on layers: {self.layer_indices}")

    def clear(self):
        self.activations.clear()

    def remove_all(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.activations.clear()
