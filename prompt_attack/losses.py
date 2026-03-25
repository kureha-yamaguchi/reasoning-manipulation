"""Loss functions for GCG+IRIS attacks."""

import torch
from torch import Tensor


def token_forcing_loss(logits: Tensor, target_ids: Tensor) -> Tensor:
    """Cross-entropy loss forcing model to produce target tokens.

    Args:
        logits: [batch, seq_len, vocab_size] — logits aligned to target positions.
        target_ids: [batch, seq_len] — target token IDs.

    Returns:
        [batch] — mean CE loss per sample.
    """
    return (
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction="none",
        )
        .reshape(logits.shape[0], -1)
        .mean(dim=-1)
    )


def refusal_direction_loss(activations: Tensor, refusal_vec: Tensor,
                           start: int, end: int) -> Tensor:
    """Raw dot-product squared between activations and refusal direction.

    Args:
        activations: [batch, seq_len, hidden_dim]
        refusal_vec: [hidden_dim] — unit normalized refusal direction.
        start: Start position (inclusive) for the region of interest.
        end: End position (exclusive) for the region of interest.

    Returns:
        [batch] — mean dot_product^2 over the specified token positions.
    """
    region = activations[:, start:end]  # [batch, n_tokens, hidden_dim]
    dot_products = torch.matmul(region, refusal_vec)  # [batch, n_tokens]
    return dot_products.pow(2).mean(dim=-1)  # [batch]


def multi_layer_refusal_loss(
    layer_activations: dict[int, Tensor],
    refusal_vec: Tensor,
    start: int,
    end: int,
) -> Tensor:
    """IRIS loss: sum of mean (r̂ᵀh)² over a token range across all layers.

    For each layer, computes the mean squared dot product between the refusal
    direction and the hidden states at positions [start, end), then sums
    across layers. This captures refusal signal in both the input (suffix)
    and the target (first CoT tokens) positions.

    Args:
        layer_activations: {layer_idx: [batch, seq_len, hidden_dim]} from hooks.
        refusal_vec: [hidden_dim] — unit normalized refusal direction.
        start: Start position (inclusive).
        end: End position (exclusive).

    Returns:
        [batch] — sum across layers of mean dot_product^2 over token range.
    """
    loss = None
    for layer_idx in sorted(layer_activations.keys()):
        act = layer_activations[layer_idx]  # [batch, seq_len, hidden_dim]
        region = act[:, start:end, :]  # [batch, n_tokens, hidden_dim]
        dots = torch.matmul(region, refusal_vec)  # [batch, n_tokens]
        layer_loss = dots.pow(2).mean(dim=-1)  # [batch]
        if loss is None:
            loss = layer_loss
        else:
            loss = loss + layer_loss
    return loss


def combined_loss(token_loss: Tensor, refusal_loss: Tensor, beta: float) -> Tensor:
    """Weighted combination: L = (1-β)*token_loss + β*refusal_loss.

    Matches the IRIS paper (Eq. 8) — raw weighted sum, no normalization.

    Args:
        token_loss: [batch] — token forcing CE loss.
        refusal_loss: [batch] — IRIS refusal direction loss.
        beta: Weight for refusal loss (0.0 = pure GCG, 1.0 = pure IRIS).

    Returns:
        [batch] — combined loss.
    """
    return (1.0 - beta) * token_loss + beta * refusal_loss
