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
    """Cosine similarity squared between activations and refusal direction.

    Args:
        activations: [batch, seq_len, hidden_dim]
        refusal_vec: [hidden_dim] — unit normalized refusal direction.
        start: Start position (inclusive) for the region of interest.
        end: End position (exclusive) for the region of interest.

    Returns:
        [batch] — mean cos_sim^2 over the specified token positions.
    """
    region = activations[:, start:end]  # [batch, n_tokens, hidden_dim]
    # Cosine similarity: dot(act, ref) / (||act|| * ||ref||)
    # refusal_vec is already unit-normalized, so just need ||act||
    act_norm = region.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [batch, n_tokens, 1]
    cos_sim = torch.matmul(region, refusal_vec) / act_norm.squeeze(-1)  # [batch, n_tokens]
    return cos_sim.pow(2).mean(dim=-1)  # [batch]


def combined_loss(token_loss: Tensor, refusal_loss: Tensor, beta: float) -> Tensor:
    """Weighted combination: (1-beta)*token_loss + beta*refusal_loss.

    Args:
        token_loss: [batch] — token forcing CE loss.
        refusal_loss: [batch] — IRIS refusal direction loss.
        beta: Weight for refusal loss (0.0 = pure GCG, 1.0 = pure IRIS).

    Returns:
        [batch] — combined loss.
    """
    return (1.0 - beta) * token_loss + beta * refusal_loss
