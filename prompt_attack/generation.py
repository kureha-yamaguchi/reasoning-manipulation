"""Extended generation utilities for GCG+IRIS attacks."""

import torch
from torch import Tensor


def generate_extended_tokens(
    model,
    embed_layer: torch.nn.Embedding,
    input_embeds: Tensor,
    n_tokens: int,
    past_key_values=None,
) -> Tensor:
    """Generate additional tokens via greedy decoding, appending embeddings.

    Works for any batch size. Uses KV cache from the initial forward pass so
    each additional token is O(1) rather than O(seq_len).

    The returned tensor includes the original input_embeds concatenated with
    the generated token embeddings, maintaining gradient flow from input_embeds.

    Args:
        model: The language model.
        embed_layer: Model's embedding layer.
        input_embeds: [batch, seq_len, hidden_dim] — initial embeddings.
        n_tokens: Number of additional tokens to generate.
        past_key_values: Optional KV cache for prefix (from before input_embeds).

    Returns:
        extended_embeds: [batch, seq_len + n_tokens, hidden_dim]
    """
    if n_tokens <= 0:
        return input_embeds

    extended = input_embeds
    output = model(
        inputs_embeds=input_embeds,
        past_key_values=past_key_values,
        use_cache=True,
    )
    kv_cache = output.past_key_values

    for _ in range(n_tokens):
        next_id = output.logits[:, -1:, :].argmax(dim=-1)  # [batch, 1]
        next_embed = embed_layer(next_id)  # [batch, 1, hidden_dim]
        extended = torch.cat([extended, next_embed], dim=1)

        output = model(
            inputs_embeds=next_embed,
            past_key_values=kv_cache,
            use_cache=True,
        )
        kv_cache = output.past_key_values

    return extended
