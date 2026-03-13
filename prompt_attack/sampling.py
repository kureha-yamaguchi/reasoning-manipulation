"""Candidate sampling and filtering for GCG optimization."""

from typing import Optional

import torch
import transformers
from torch import Tensor


def sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Optional[Tensor] = None,
) -> Tensor:
    """Sample candidate token sequences based on gradient information.

    Args:
        ids: Current optimized token IDs [seq_len].
        grad: Token gradient [seq_len, vocab_size].
        search_width: Number of candidate sequences.
        topk: Top-k tokens to consider per position.
        n_replace: Tokens to replace per candidate.
        not_allowed_ids: Token IDs to exclude.

    Returns:
        [search_width, seq_len] candidate token sequences.
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(
        torch.rand((search_width, n_optim_tokens), device=grad.device)
    )[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)
    return new_ids


def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer) -> Tensor:
    """Filter candidates that change after retokenization."""
    ids_decoded = tokenizer.batch_decode(ids)
    filtered = []
    for i in range(len(ids_decoded)):
        ids_encoded = tokenizer(
            ids_decoded[i], return_tensors="pt", add_special_tokens=False
        ).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
            filtered.append(ids[i])

    if not filtered:
        raise RuntimeError(
            "No token sequences survived retokenization filtering. "
            "Try a different optim_str_init or set filter_ids=False."
        )
    return torch.stack(filtered)
