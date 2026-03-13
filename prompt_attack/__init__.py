"""GCG+IRIS attack: optimizes adversarial suffixes to suppress refusal directions in CoT reasoning."""

from prompt_attack.config import GCGConfig as GCGConfig
from prompt_attack.gcg import GCG as GCG
from prompt_attack.gcg import GCGResult as GCGResult
from prompt_attack.gcg import run as run
from prompt_attack.models import ModelConfig as ModelConfig
from prompt_attack.models import get_model_config as get_model_config
