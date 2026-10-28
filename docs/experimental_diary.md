# Experimental diary

Decisions, rationale and outcomes, newest last. Code/data provenance lives in run metadata
(`results/<model>/dataset/quadrants/run_metadata/*.json`) and git history.

## 2026-10-28 — ICML resubmission: extend Fig 3 (quadrant prefix resampling) to DeepSeek-R1-Distill-Qwen-7B

**Goal.** Breadth before method: reproduce Fig 3 for a third model before changing the
method (e.g. Thought-Anchors-style counterfactual importance). R1-Distill-Qwen-7B shares
the teacher and recipe with R1-Distill-Llama-8B but not the base model, so it tests whether
"early commitment" is a property of R1 distillation or of Llama.

**Reproducibility audit of Fig 3 (before any new runs).**
- `plot_quadrant_matrix.py` drew the N=32 line-plot subset and the heatmap prompts with
  unseeded `np.random.choice`. Llama has exactly 32 eligible-and-rolled prompts, so Fig 3a's
  line plot is reproducible; GPT-OSS (38 rolled) and both heatmaps were not. Fixed: `--seed`,
  `--heatmap_prompts`, and a `_prompts.json` sidecar. Paper heatmaps reproduce with
  `--heatmap_prompts 806,963,1073` / `571,611,2057`.
- Fig 3 caption says the band is "±1 SD across prompts"; the code computes it across
  (prompt, CoT) rows. Caption should be corrected.
- Quadrant selection protocol (verified against existing rollouts): three scored CSVs
  (train_harmful, orbench_extra, test_harmful), x_threshold 0.03, y_threshold 0.43
  (strict `>`). Under this protocol all 32 Llama rollout prompts match `find_quadrant`
  indices. `resample_quadrants.py` argparse default was 0.4 (now 0.43).
- Llama's paper set is not exactly `random.seed(42)` sample of its 33 eligible prompts
  (has 1390 where the seeded sample has 335): original sampling not fully reproducible.
- The `test_harmful` scored CSVs for Qwen-7B and GPT-OSS had been deleted from git; restored
  from history (8bb53fb^, d561243^). Qwen-7B's contains one `'nan'` score which crashed
  `find_quadrant` (likely why the earlier 8 Qwen-7B prompts were selected from 2 CSVs).
  Loader now skips non-finite scores. Prompt indices are identical with 2 or 3 CSVs (test
  prompts are appended), so the 8 existing Qwen-7B rollouts remain valid.
- Judge: StrongREJECT rubric with local GPT-OSS-20B via vLLM (`STRONGREJECT_VLLM_URL`),
  max-model-len 8192. **Caveat for the paper:** GPT-OSS-20B judges its own outputs in the
  GPT-OSS arm; a second judge on a subset would pre-empt this.

**Decision: roll out all 34 eligible Qwen-7B prompts** (8 existed, 26 new) instead of a
random 32, removing sampling ambiguity. Also roll out Llama's one missing eligible prompt
(335) so Llama covers all 33.

**Generation changes** (`resample_quadrants.py`, commit 45607bf): all (CoT, depth, rep)
requests for a prompt in one `llm.generate` call; per-request seeds; per-CoT resume and
atomic writes; `--shard k/K`; run metadata. Prompts are byte-identical to the old per-depth
loop (checked on Qwen-7B prompt 287, all 5 CoTs). Throughput ~10k output tok/s per H100,
~8 min per prompt per GPU. Model weights read from /scratch (Weka; load 7 s vs 417 s from
NFS); results written to /data. As in the original code, a depth occasionally yields 0/10
valid rollouts (no `</think>` within 16k tokens): 2/2023 depths in the old Qwen-7B data,
1 so far in the new run.

**Commitment estimands** (`utils/heuristic/commitment.py`; existing data, cluster bootstrap
over prompts, 10k resamples, 95% CI):

| | Llama-8B (32 prompts) | GPT-OSS-20B (38) |
|---|---|---|
| first entry, delta=0.2 | 0.05 [0.04, 0.07] | 0.14 [0.12, 0.17] |
| last exit, delta=0.2 | 0.22 [0.17, 0.28] | 0.24 [0.21, 0.28] |

Llama *reaches* its final outcome much earlier, but is not more irrevocably committed: it
has a persistent residual std (~0.05–0.1) late in the CoT, while GPT-OSS declines gradually
to ~0 (curves cross at ~0.4; `utils/heuristic/plot_std_overlay.py`). The paper's "decided in
the first few sentences" claim holds for first entry, not last exit; report both.
