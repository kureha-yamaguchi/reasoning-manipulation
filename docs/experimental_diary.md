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

**Tokens vs sentences.** Quadrant CoTs, median tokens (sentences; tokens/sentence):
Llama 607 (37; 15.4), Qwen 768 (48; 16.0), GPT-OSS 233 (19; 12.6). So the length
confound holds in tokens; first entry ≈ 29 / 52 / 35 tokens (sentences × tokens/sentence).

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

### GPT-OSS extended to all 94 eligible prompts (2026-10-29, 00:39–03:38)

57 new prompts (3 H100s, ~70 min generation + ~100 min scoring with 2 judges). The gpt-oss
chat template inserts today's date into the system prompt; existing rollouts mixed four
dates (2026-03-23/24, 04-18/19). New runs pin `--system_date 2026-04-18` (modal existing
date); with it the batched generator reproduces existing prompts exactly. Legacy prompt 464
(eligible only at y=0.4, not 0.43) is excluded: analyses now restrict each model to the
prompts in its latest `run_metadata` sample = full eligible set (33 / 34 / 94).
NaN repair: 539/565 recovered.

**Final (δ=0.2; 95% cluster bootstrap over prompts)** — supersedes GPT-OSS(38) above:

| | Llama-8B (33) | Qwen-7B (34) | GPT-OSS-20B (94) |
|---|---|---|---|
| first entry, normalised | 0.050 [0.038, 0.064] | 0.071 [0.054, 0.089] | 0.142 [0.128, 0.157] |
| first entry, sentences | 1.86 [1.24, 2.62] | 3.23 [2.22, 4.40] | 2.72 [2.48, 2.97] |
| last exit, normalised | 0.215 [0.161, 0.274] | 0.322 [0.249, 0.398] | 0.235 [0.217, 0.254] |

Differences: normalised first entry GPT-OSS − Llama +0.092 [+0.072, +0.112]*, GPT-OSS − Qwen
+0.072 [+0.048, +0.094]*, Qwen − Llama +0.021 [−0.001, +0.043]. Sentences: Qwen − Llama
+1.37 [+0.12, +2.68]*, GPT-OSS − Llama +0.86 [+0.05, +1.55]*, GPT-OSS − Qwen −0.51
[−1.68, +0.52]. Last exit: Qwen − Llama +0.107*, GPT-OSS − Qwen −0.087*, GPT-OSS − Llama
+0.020 n.s.

GPT-OSS estimates with 94 prompts match the 38-prompt subset (first entry 0.142 vs 0.142,
last exit 0.235 vs 0.242) with ~half the CI width: the old subset was not biased.
The absolute-sentence overlay (`figures/std_overlay_rep10_sentences25.pdf`) is the clearest
statement of the result: GPT-OSS resolves over ~5 sentences then locks in (std ≈ 0.05);
both R1 distills drop faster at S1 but plateau at std ≈ 0.10 (Llama) / 0.15–0.17 (Qwen).

## 2026-10-29 — R1 (teacher) arm queued; Thought-Anchors counterfactual importance demo

**DeepSeek-R1 arm.** Full R1 cannot run on a 3/4×H100 node; 4-bit AWQ (`QuixiAI/DeepSeek-R1-AWQ`
@21fb456e, 365 GB) fits one 8×H100 node (htc-g059/060). Pipeline `bash_scripts/r1_quadrants.sbatch`
(Slurm job 9207461): candidate pool → cot5_out5 baseline → gpt-oss judge → quadrant rollouts, each
phase resumable. Decisions: (i) candidate pool = union of prompts quadrant-eligible for any of the
3 models (159 prompts; only 2 shared between models — eligible sets are nearly disjoint), then
R1's own thresholds select from it, rather than a full 2,446-prompt baseline; (ii) the AWQ repo
ships R1's original template (no `<think>\n`); we use the official R1 tokenizer @56d4cbbb (same
vocab, template forces `<think>\n` as the distills' do). Caveat: 4-bit quant ≠ FP8 teacher.
Smoke-tested end to end with Qwen-7B on 1 GPU (sparse worktree, loosened thresholds).

**Counterfactual importance (Bogdan et al. 2025, §3.2) from existing rollouts** —
`utils/heuristic/counterfactual_importance.py`. The first sentence generated at depth i is a
resample T_i of S_i, so no new generation is needed. Definitions as the paper (KL[p(A'|T_i≉S_i) ||
p(A|S_i kept)]; dissimilar = cosine < median over all pairs, all-MiniLM-L6-v2). Adaptations:
k=10 not 100; A binarised at score 0.5; Laplace α=1; per-sentence permutation test (2,000 shuffles).

Demo, Llama-8B, paper heatmap prompts 806/963/1073 (15 CoTs, 508 sentences, 4,873 pairs):
- Similarity quartiles 0.27/0.47/0.71 (paper's median: 0.8), so threshold = 0.466.
- **Resampling importance is concentrated at S_0**: mean 0.50 (median 0.49) vs ≤0.04 for S_1
  and <0.01 from S_3 on; S_0 = 71% of total RI. Consistent with Fig 3's first-sentence drop, now
  attributed to a specific sentence.
- **The embedding filter fails exactly where it matters**: 13/15 S_0 have no "dissimilar"
  resample. Openings restate the prompt, so MiniLM rates them similar, yet they differ in stance
  ("I need to create a fake news story…" → comply; "the user is asking me to…" / "how to respond
  to this request" → refuse). Topic-level similarity misses stance.
- Counterfactual importance beyond S_0 is noise at this n (2–5 dissimilar resamples); 2/508
  sentences pass p<0.05 (both S_0), vs ~20 expected by chance — the test is conservative/underpowered.

**Next (proposed):** (1) stance-aware equivalence for T_i vs S_i (LLM judge with local gpt-oss, or
cluster openings by stance) instead of cosine; (2) targeted resampling with ~100 rollouts at the
first ~5 sentences of a CoT subset (Llama ~15 min on 1 H100) for paper-level power; (3) run all
prompts and models. Selection caveat: quadrant prompts are chosen for high across-CoT variance,
so they are enriched for early forks.

### Dense resampling + judge equivalence (Llama-8B, 2026-10-29 evening)

100 fresh rollouts at depths 0–5 for the 15 CoTs of prompts 806/963/1073 (`dense_resample`, seed
base 1000; 9,000 rollouts, all valid), scored with the gpt-oss judge; (S_i, T_i) equivalence for
i ≤ 4 labelled SAME/DIFFERENT by gpt-oss-20b (`sentence_equivalence`, 3,991 unique pairs).

- **S_0 importance is robust at n=100**: resampling importance significant (permutation p<0.05)
  for 14/15 CoTs; mean 0.98 vs 0.04 (S_1), ≤0.01 (S_2–S_4).
- **Neither filter isolates what S_0 carries.** Judge: only 7.7% of S_0 resamples DIFFERENT
  (cosine median rule: median 3/100). Resamples judged SAME have comply rates at each prompt's base
  rate (~0.2 / 0.75 / 0.55 for 806 / 963 / 1073), irrespective of the kept S_0's outcome (0.0–0.98).
  So either the outcome hinges on fine wording of S_0, or the judge is too lenient — it is at least
  partly the latter: "I came across this really concerning question…" was labelled SAME as
  "I'm trying to figure out how someone might commit identity theft…".
- Agreement cosine vs judge is weak (of 3,750 cosine-different pairs, judge calls 73% SAME).

**Next (proposed):** replace the pairwise SAME/DIFFERENT prompt with a stance taxonomy for openings
(e.g. adopts task / describes request neutrally / flags concern / plans safe alternative), validated
on a small hand-labelled set; then (a) P(comply | stance of T_0) as a direct test of the stance
hypothesis and (b) "different" := stance category differs. This is also mythos's hinge taxonomy.
