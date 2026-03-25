"""Core GCG+IRIS optimizer."""

import atexit
import copy
import gc
import logging
from dataclasses import asdict
from typing import List, Optional, Union

import torch
import transformers
import wandb
from torch import Tensor
from tqdm import tqdm
from transformers import set_seed
from transformers.cache_utils import DynamicCache

from prompt_attack.buffer import AttackBuffer
from prompt_attack.config import GCGConfig
from prompt_attack.hooks import ActivationHookManager
from prompt_attack.losses import combined_loss, multi_layer_refusal_loss, token_forcing_loss
from prompt_attack.sampling import filter_ids, sample_ids_from_grad
from prompt_attack.utils import (
    INIT_CHARS,
    find_executable_batch_size,
    get_nonascii_toks,
)

logger = logging.getLogger("gcg-iris")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class GCGResult:
    def __init__(self, best_loss: float, best_string: str, losses: List[float],
                 strings: List[str], best_answer: Optional[str] = None):
        self.best_loss = best_loss
        self.best_string = best_string
        self.losses = losses
        self.strings = strings
        self.best_answer = best_answer


class GCG:
    """GCG+IRIS optimizer: optimizes an adversarial suffix to minimize a combined
    token-forcing + refusal-direction loss.

    IRIS loss (Eq. 8, Huang et al. NAACL 2025): computed on the last INPUT
    token's hidden state across ALL layers, not on output/generated tokens.
    """

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
        refusal_vector: Optional[Tensor] = None,
        refusal_vector_path: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = (
            None if config.allow_non_ascii
            else get_nonascii_toks(tokenizer, device=model.device)
        )

        self.prefix_cache = None
        self.hook_manager: Optional[ActivationHookManager] = None
        self.refusal_vec: Optional[Tensor] = None
        self.step = 0
        self.wandb_metrics: dict = {}
        self.using_wandb = False
        self.stop_flag = False

        # Setup IRIS if enabled
        if config.use_iris and config.beta > 0.0:
            self._setup_iris(refusal_vector, refusal_vector_path)

        # Setup wandb
        if config.wandb_config:
            if wandb.run is not None:
                wandb.finish()
            self.using_wandb = True
            wandb_cfg = asdict(config)
            # Merge any extra config (e.g. model name) from wandb_config
            extra = config.wandb_config.get("config", {})
            wandb_cfg.update(extra)
            wandb.init(
                project=config.wandb_config.get("project", "gcg-iris"),
                entity=config.wandb_config.get("entity"),
                config=wandb_cfg,
                tags=config.wandb_config.get("tags"),
                group=config.wandb_config.get("group"),
            )
            if wandb.run is not None:
                name = config.wandb_config.get("name")
                if name:
                    wandb.run.name = name
                logger.info(f"Initialized wandb run: {wandb.run.name}")
        else:
            logger.info("wandb not configured.")

        atexit.register(self._cleanup)

    def _setup_iris(self, refusal_vector: Optional[Tensor], refusal_vector_path: Optional[str]):
        """Load refusal vector and register activation hooks on ALL layers."""
        if refusal_vector is not None:
            self.refusal_vec = refusal_vector.to(self.model.device, self.model.dtype)
        elif refusal_vector_path:
            self.refusal_vec = torch.load(refusal_vector_path, map_location=self.model.device)
            self.refusal_vec = self.refusal_vec.to(dtype=self.model.dtype)
            logger.info(f"Loaded refusal vector from {refusal_vector_path}")
        else:
            raise ValueError("IRIS enabled but no refusal_vector or refusal_vector_path provided")

        # Unit-normalize
        self.refusal_vec = self.refusal_vec / self.refusal_vec.norm()

        # Register hooks on ALL layers (IRIS paper: sum across all layers)
        num_layers = len(self.model.model.layers)
        all_layers = list(range(num_layers))
        self.hook_manager = ActivationHookManager(self.model, all_layers)
        logger.info(f"IRIS: hooked all {num_layers} layers for refusal loss")

    def _cleanup(self):
        if self.using_wandb and wandb.run is not None:
            wandb.finish()
        if self.hook_manager:
            self.hook_manager.remove_all()

    # ── Assembly ─────────────────────────────────────────────────────

    def _assemble_embeds(
        self,
        optim_embeds: Tensor,
        batch_size: int = 1,
    ) -> Tensor:
        """Assemble full input embeddings: [before?] + optim + after + target.

        When prefix_cache is used, before_embeds is omitted (already in cache).
        """
        after = self.after_embeds.expand(batch_size, -1, -1)
        target = self.target_embeds.expand(batch_size, -1, -1)

        if self.prefix_cache:
            return torch.cat([optim_embeds, after, target], dim=1)
        else:
            before = self.before_embeds.expand(batch_size, -1, -1)
            return torch.cat([before, optim_embeds, after, target], dim=1)

    def _expand_cache(self, batch_size: int) -> Optional[DynamicCache]:
        """Expand prefix KV cache for batched inference."""
        if not self.prefix_cache:
            return None

        expanded = DynamicCache()
        for layer_idx in range(len(self.prefix_cache)):
            # Use new API if available, fall back to deprecated
            if hasattr(self.prefix_cache, 'layers'):
                k = self.prefix_cache.layers[layer_idx].keys.expand(batch_size, -1, -1, -1)
                v = self.prefix_cache.layers[layer_idx].values.expand(batch_size, -1, -1, -1)
            else:
                k = self.prefix_cache.key_cache[layer_idx].expand(batch_size, -1, -1, -1)
                v = self.prefix_cache.value_cache[layer_idx].expand(batch_size, -1, -1, -1)
            expanded.update(k, v, layer_idx)
        return expanded

    def _iris_token_range(self, input_embeds: Tensor) -> tuple[int, int]:
        """Token range for IRIS loss: from suffix start to end of target tokens.

        Covers the adversarial suffix + template tokens + target CoT tokens.
        With prefix cache: [optim, after, target] — starts at 0.
        Without prefix cache: [before, optim, after, target] — starts after before.
        """
        if self.prefix_cache:
            start = 0  # suffix starts at beginning (before is in cache)
        else:
            start = self.before_embeds.shape[1]  # skip before tokens
        end = input_embeds.shape[1]  # through end of target
        return start, end

    # ── Core optimization ────────────────────────────────────────────

    def run(
        self,
        messages: Union[str, List[dict]],
        target: str,
    ) -> GCGResult:
        """Run the GCG+IRIS optimization loop.

        Args:
            messages: User message(s).
            target: Target string (first N tokens of ortho CoT).

        Returns:
            GCGResult with best loss, suffix, and trajectories.
        """
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(False)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)

        # Insert {optim_str} placeholder as suffix
        if not any("{optim_str}" in d["content"] for d in messages):
            messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        template = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self.tokenizer.bos_token and template.startswith(self.tokenizer.bos_token):
            template = template.replace(self.tokenizer.bos_token, "", 1)

        before_str, after_str = template.split("{optim_str}")
        if config.add_space_before_target:
            target = " " + target

        # Tokenize fixed parts
        before_ids = self.tokenizer(
            [before_str], padding=False, return_tensors="pt"
        )["input_ids"].to(self.model.device, torch.int64)
        after_ids = self.tokenizer(
            [after_str], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.model.device, torch.int64)
        self.target_ids = self.tokenizer(
            [target], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.model.device, torch.int64)

        # Embed fixed parts
        self.before_embeds = self.embedding_layer(before_ids)
        self.after_embeds = self.embedding_layer(after_ids)
        self.target_embeds = self.embedding_layer(self.target_ids)

        # Build prefix cache
        if config.use_prefix_cache:
            with torch.no_grad():
                out = self.model(inputs_embeds=self.before_embeds, use_cache=True)
                if isinstance(out.past_key_values, tuple):
                    self.prefix_cache = DynamicCache.from_legacy_cache(out.past_key_values)
                else:
                    self.prefix_cache = out.past_key_values

        # Initialize buffer
        buffer = self._init_buffer()
        optim_ids = buffer.get_best_ids()

        losses: list[float] = []
        optim_strings: list[str] = []
        prompt_string = messages[-1]["content"].replace("{optim_str}", "")

        for step in tqdm(range(config.num_steps)):
            self.step = step

            # 1) Compute gradient (batch=1)
            grad = self._compute_gradient(optim_ids)

            with torch.no_grad():
                # 2) Sample candidates
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0), grad.squeeze(0),
                    config.search_width, config.topk, config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )
                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, self.tokenizer)

                n_candidates = sampled_ids.shape[0]

                # 3) Evaluate candidates
                cand_embeds = self._assemble_embeds(
                    self.embedding_layer(sampled_ids), batch_size=n_candidates
                )
                batch_size = n_candidates if config.batch_size is None else config.batch_size
                loss = find_executable_batch_size(
                    self._evaluate_candidates, batch_size
                )(cand_embeds)

                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = self.tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            buffer.log_buffer(self.tokenizer, logger)
            self._log_step(step, current_loss, buffer, prompt_string)

            if self.stop_flag:
                logger.info("Early stopping.")
                break

        # Final generation
        best_str = self.tokenizer.batch_decode(buffer.get_best_ids())[0]
        full_prompt = prompt_string + best_str
        response = self.generate_response(full_prompt)

        if self.using_wandb and wandb.run:
            wandb.run.summary["best_answer"] = response
            wandb.finish()

        min_idx = losses.index(min(losses))
        return GCGResult(
            best_loss=losses[min_idx],
            best_string=optim_strings[min_idx],
            losses=losses,
            strings=optim_strings,
            best_answer=response,
        )

    # ── Gradient computation ─────────────────────────────────────────

    def _compute_gradient(self, optim_ids: Tensor) -> Tensor:
        """Compute gradient of combined loss w.r.t. one-hot token matrix.

        IRIS loss is computed over suffix+target token positions across all
        layers. No extended generation needed — everything is in the forward pass.
        """
        if self.hook_manager:
            self.hook_manager.clear()

        # One-hot -> embeddings (differentiable)
        onehot = torch.nn.functional.one_hot(
            optim_ids, num_classes=self.embedding_layer.num_embeddings
        ).to(self.model.device, self.model.dtype)
        onehot.requires_grad_()
        optim_embeds = onehot @ self.embedding_layer.weight

        input_embeds = self._assemble_embeds(optim_embeds, batch_size=1)

        # Single forward pass
        if self.prefix_cache:
            out = self.model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache)
        else:
            out = self.model(inputs_embeds=input_embeds)

        logits = out.logits

        # Token forcing loss (over target token positions)
        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        target_logits = logits[:, shift - 1 : shift - 1 + self.target_ids.shape[1], :]
        t_loss = token_forcing_loss(target_logits, self.target_ids)

        if self.using_wandb:
            self.wandb_metrics["token_loss"] = t_loss.mean().item()

        # IRIS loss: suffix through target tokens, across all layers
        use_iris = self.config.use_iris and self.config.beta > 0.0 and self.refusal_vec is not None
        if use_iris and self.hook_manager and self.hook_manager.activations:
            iris_start, iris_end = self._iris_token_range(input_embeds)
            r_loss = multi_layer_refusal_loss(
                self.hook_manager.activations, self.refusal_vec, iris_start, iris_end,
            )

            loss = combined_loss(t_loss, r_loss, self.config.beta)

            if self.using_wandb:
                self.wandb_metrics["refusal_loss"] = r_loss.mean().item()
                self.wandb_metrics["combined_loss"] = loss.mean().item()
                self.wandb_metrics["n_layers_hooked"] = len(self.hook_manager.activations)
        else:
            loss = t_loss

        grad = torch.autograd.grad(outputs=[loss], inputs=[onehot])[0]
        return grad

    # ── Candidate evaluation ─────────────────────────────────────────

    def _evaluate_candidates(self, search_batch_size: int, input_embeds: Tensor) -> Tensor:
        """Evaluate candidate suffixes.

        Single forward pass per batch — IRIS loss over suffix+target token
        range across all layers. No extended generation needed.
        """
        all_loss = []
        use_iris = self.config.use_iris and self.config.beta > 0.0 and self.refusal_vec is not None
        prefix_cache_batch = None

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                if self.hook_manager:
                    self.hook_manager.clear()

                batch = input_embeds[i : i + search_batch_size]
                bs = batch.shape[0]

                if self.prefix_cache:
                    if prefix_cache_batch is None or bs != search_batch_size:
                        prefix_cache_batch = self._expand_cache(bs)
                    out = self.model(inputs_embeds=batch, past_key_values=prefix_cache_batch)
                else:
                    out = self.model(inputs_embeds=batch)

                logits = out.logits

                # Token forcing loss
                shift = input_embeds.shape[1] - self.target_ids.shape[1]
                target_logits = logits[:, shift - 1 : shift - 1 + self.target_ids.shape[1], :]
                target_labels = self.target_ids.expand(bs, -1)
                t_loss = token_forcing_loss(target_logits, target_labels)

                # IRIS loss: suffix through target, across all layers
                if use_iris and self.hook_manager and self.hook_manager.activations:
                    iris_start, iris_end = self._iris_token_range(input_embeds)
                    r_loss = multi_layer_refusal_loss(
                        self.hook_manager.activations, self.refusal_vec,
                        iris_start, iris_end,
                    )
                    batch_loss = combined_loss(t_loss, r_loss, self.config.beta)
                else:
                    batch_loss = t_loss

                all_loss.append(batch_loss)

                del out
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

    # ── Buffer initialization ────────────────────────────────────────

    def _init_buffer(self) -> AttackBuffer:
        """Initialize the attack buffer with initial optim string."""
        buffer = AttackBuffer(self.config.buffer_size)

        if isinstance(self.config.optim_str_init, str):
            init_ids = self.tokenizer(
                self.config.optim_str_init, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(self.model.device)

            if self.config.buffer_size > 1:
                char_ids = self.tokenizer(
                    INIT_CHARS, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].squeeze().to(self.model.device)
                n_extra = self.config.buffer_size - 1
                rand_indices = torch.randint(0, char_ids.shape[0], (n_extra, init_ids.shape[1]))
                init_ids = torch.cat([init_ids, char_ids[rand_indices]], dim=0)
        else:
            init_ids = self.tokenizer(
                self.config.optim_str_init, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(self.model.device)

        true_size = max(1, self.config.buffer_size)
        init_embeds = self._assemble_embeds(
            self.embedding_layer(init_ids), batch_size=init_ids.shape[0]
        )
        init_losses = find_executable_batch_size(
            self._evaluate_candidates, true_size
        )(init_embeds)

        for j in range(init_ids.shape[0]):
            buffer.add(init_losses[j].item(), init_ids[[j]])

        buffer.log_buffer(self.tokenizer, logger)
        return buffer

    # ── Generation ───────────────────────────────────────────────────

    def generate_response(self, prompt: str) -> str:
        """Generate a full response for evaluation."""
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device, torch.int64)

        output = self.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(
            output[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0]

    # ── Logging ──────────────────────────────────────────────────────

    def _log_step(self, step, current_loss, buffer, prompt_string):
        if not self.using_wandb:
            return

        self.wandb_metrics["loss"] = current_loss
        self.wandb_metrics["best_loss"] = buffer.get_lowest_loss()
        self.wandb_metrics["step"] = step
        self.wandb_metrics["beta"] = self.config.beta

        # Periodic full generation
        if step % self.config.eval_frequency == 0:
            best_str = self.tokenizer.batch_decode(buffer.get_best_ids())[0]
            full_prompt = prompt_string + best_str
            response = self.generate_response(full_prompt)
            if wandb.run:
                wandb.run.summary["best_answer"] = response
                wandb.run.summary["best_string"] = best_str

        wandb.log(self.wandb_metrics)
        self.wandb_metrics = {}


# ── Public API ─────────────────────────────────────────────────────────

def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    config: Optional[GCGConfig] = None,
    refusal_vector: Optional[Tensor] = None,
    refusal_vector_path: Optional[str] = None,
) -> GCGResult:
    """Run GCG+IRIS optimization.

    Args:
        model: Target model.
        tokenizer: Model tokenizer.
        messages: User message(s).
        target: Target CoT string.
        config: GCG configuration.
        refusal_vector: Pre-loaded refusal direction tensor.
        refusal_vector_path: Path to refusal direction .pt file.

    Returns:
        GCGResult with optimized suffix and metrics.
    """
    if config is None:
        config = GCGConfig()

    logger.setLevel(getattr(logging, config.verbosity))

    gcg = GCG(model, tokenizer, config, refusal_vector, refusal_vector_path)
    return gcg.run(messages, target)
