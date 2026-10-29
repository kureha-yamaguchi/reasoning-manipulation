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

### Results — Qwen-7B added (2026-10-29)

**Data.** Qwen-7B: all 34 eligible prompts, 162 CoT files, 81,459 scored rollouts (26 new
prompts generated 20:53–22:50 on 3 H100s). Llama: all 33 eligible (added 335). GPT-OSS:
unchanged 38 prompts. NaN rate (judge context overflow at 8192 tokens, or parse failure):
Llama 2.18%, Qwen 1.25% before / 0.41% after repair (611/808 rows recovered; the rest
are responses longer than the judge context), GPT-OSS 0.70%. NaNs drop out of per-depth
means, as in the paper. Repair changed Qwen first entry 0.070 → 0.071 (tables below are
post-repair; differences computed pre-repair, unchanged to 2 d.p.).
Rollout CSVs are gitignored (`results/*/*/dataset*`), stored on /data only.

Scoring throughput fix: `strong_reject.evaluate_dataset` spawns cpu_count processes per file
→ CPU-bound at ~5 rows/s. Threaded calls to the same `strongreject_rubric` + one gpt-oss
server per GPU → ~21 rows/s. vLLM 0.10.1 `--data-parallel-size` crashes on gpt-oss MoE.
Judge test–retest on 60 rows: 98% binarised agreement, mean |diff| 0.038.

**Commitment (δ=0.2; mean over prompts, 95% cluster bootstrap)** — `results/commitment/`:

| | Llama-8B (33) | Qwen-7B (34) | GPT-OSS-20B (38) |
|---|---|---|---|
| first entry, normalised | 0.050 [0.038, 0.064] | 0.071 [0.054, 0.089] | 0.142 [0.117, 0.170] |
| first entry, sentences | 1.9 [1.2, 2.6] | 3.2 [2.2, 4.4] | 2.8 [2.3, 3.3] |
| last exit, normalised | 0.215 [0.161, 0.274] | 0.322 [0.249, 0.398] | 0.242 [0.209, 0.278] |
| CoT length (sentences, median [IQR]) | 37 [23, 56] | 48 [24, 67] | 20 [13, 31] |

Differences (bootstrap CI of difference; * = excludes 0; 18 comparisons, uncorrected):
- Normalised first entry: GPT-OSS later than both distills at every δ (+0.07 to +0.12 *);
  Qwen − Llama +0.02 [−0.00, +0.04].
- Absolute first entry (sentences): Qwen − Llama +1.4 [+0.1, +2.7] *; GPT-OSS − Llama
  +0.9 [+0.0, +1.7] * (marginal); GPT-OSS − Qwen −0.5 [−1.7, +0.7].
- Last exit: Qwen − Llama +0.11 [+0.02, +0.20] *; GPT-OSS − Qwen −0.08 [−0.16, −0.00] *;
  GPT-OSS − Llama n.s.

**Interpretation.**
1. *CoT-length confound.* GPT-OSS CoTs are ~half as long, so normalised position
   inflates its decision point. In absolute sentences all three models reach their final
   outcome within ~2–3 sentences; the paper's "Llama early vs GPT-OSS gradual" contrast is
   largely a units effect. Fig 3 should be accompanied by the absolute-sentence view
   (`figures/std_overlay_rep10_sentences25.pdf`).
2. *The robust distill-vs-GPT-OSS difference is residual late uncertainty*, shared by both
   R1 distills: across-rollout std plateaus at ~0.1 (Llama) / ~0.15 (Qwen) for the rest of
   the CoT, while GPT-OSS falls to ~0.05 by ~8 sentences. I.e. distilled models settle
   early but stay "re-openable"; GPT-OSS settles a few sentences later but then locks in.
   This is the more defensible "distillation signature" — it replicates across both
   same-teacher students; early first entry does not (Qwen ≈ GPT-OSS in sentences).
3. Scale (7–8B vs 20B), architecture and training paradigm remain confounded; GPT-OSS
   is judged by itself.

**Caveats.** k=10 → a single flipped rollout moves m_i by 0.1 (δ=0.1 is at the noise
floor). Absolute-sentence curves suffer survivorship beyond ~20 sentences for GPT-OSS.
GPT-OSS's 38 prompts are not a uniform sample of its 94 eligible prompts (mixed history).
δ=0.2 was fixed before looking at Qwen data; δ ∈ {0.1, 0.3} reported for sensitivity.

**Next steps (proposed).**
- Thought-Anchors-style counterfactual importance from the *existing* rollouts: the first
  sentence of `generated_text` at depth i is a resample of S_i, so semantic filtering
  needs no new generation.
- Normalise by tokens as well as sentences; check sentence-length differences.
- Second judge on a stratified subset (GPT-OSS self-judging).
- Extend GPT-OSS to all 94 eligible prompts (~56 prompts, ~4–5 GPU-hours) for a clean sample.
