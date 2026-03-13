"""GCG+IRIS attack configuration."""

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class GCGConfig:
    num_steps: int = 150
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: Optional[int] = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    seed: int = 42
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    use_prefix_cache: bool = True

    # IRIS parameters
    beta: float = 0.5
    use_iris: bool = True
    refusal_mode: str = "cot"  # "cot" or "baseline"
    refusal_layer: Optional[int] = None  # None => use best layer from model config
    extended_gen_tokens: int = 15
    target_tokens: int = 20

    # Generation
    max_new_tokens: int = 2048
    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    eval_frequency: int = 25

    # Logging
    verbosity: str = "INFO"
    wandb_config: dict = field(default_factory=dict)

    # Think tokens
    think_start_token: str = "<think>"
    think_end_token: str = "</think>"
