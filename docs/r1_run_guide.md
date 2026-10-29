# Running the Fig 3 quadrant pipeline on full-precision DeepSeek-R1

**Goal.** Add the teacher, DeepSeek-R1, as a fourth model to Fig 3: prefix-resampled rollouts on R1's
own quadrant prompts, scored with the same StrongREJECT judge as the other models. One script does
everything and can be re-run at any point. Finished work is skipped.

## Hardware

- **One node** with at least ~800 GB of GPU memory, e.g. 8×H200, 8×B200 or 8×MI300X. R1's FP8
  weights are ~690 GB, plus KV cache. The script does not support multi-node runs.
- 8×H100 (80 GB) is **not** enough for full precision. On such a node, use the default 4-bit AWQ
  model (`MODEL` unset), which is what our own cluster runs. By preference though, if compute allows then provide full precision results.
- Disk: ~700 GB for the R1 weights, plus ~26 GB for the judge (`openai/gpt-oss-20b`).
- It has only been tested on H100s with vLLM 0.10.1, which `uv.lock` pins. Blackwell or AMD GPUs may
  need a newer vLLM. If so, say which version you used.

## Setup

```bash
git clone git@github.com:kureha-yamaguchi/reasoning-manipulation.git
cd reasoning-manipulation
git checkout counterfactual-importance
uv sync
```

Both models are public on Hugging Face, so no token is needed. They download into the default HF
cache, which `HF_HOME` controls, on first use.

## 1. Smoke test (~1 h, 1 GPU), in a separate clone

This runs the full pipeline end to end on a small model with a few prompts, with loosened
thresholds so that every phase does real work. Run it in a **separate clone**: it writes under
`results/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/`, which would otherwise mix with any real results
you have there.

```bash
git clone git@github.com:kureha-yamaguchi/reasoning-manipulation.git r1-smoke && cd r1-smoke
git checkout counterfactual-importance && uv sync
CUDA_VISIBLE_DEVICES=0 MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B TP=1 CAND_LIMIT=3 N_PROMPTS=1 \
  X_THR=1 Y_THR=-1 RESAMPLE_ARGS="--max_new_tokens 2048" bash bash_scripts/r1_quadrants.sbatch
```

The run passes if it prints `Phase 1` through `Phase 5` and then `Done`, and if
`results/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/dataset/quadrants/` contains `scored_full_resample_*`
files. Delete the clone afterwards.

## 2. The real run

From the repo root, in tmux or a batch job:

```bash
MODEL=deepseek-ai/DeepSeek-R1 TP=8 bash bash_scripts/r1_quadrants.sbatch
```

- If you use Slurm, edit the `#SBATCH` lines at the top of the script (account, partition, GPUs,
  time) and `sbatch` it, with `MODEL` and `TP` exported. Plain `bash` ignores those lines.
- **If it stops** (walltime, crash, eviction), run the same command again. Each phase skips work
  that is already saved, and rollouts are written atomically, one CoT file at a time.
- Do not change the thresholds or the candidate pool (`dataset/candidates_r1/`). They must match
  the other models' runs.

**What it does:**

| Phase | Work | Rough cost (8×H200, unmeasured) |
|---|---|---|
| 1 | Copy the 159-prompt candidate pool (prompts quadrant-eligible for Llama-8B, Qwen-7B or gpt-oss-20b) | seconds |
| 2 | Baseline: 5 CoTs × 5 answers per prompt (max 2,048 new tokens, the same as the other models) | ~30–60 min |
| 3 | Score the baseline with the local gpt-oss-20b judge (1 GPU) and repair failed judge calls | ~30–60 min |
| 4 | Select R1's own quadrant prompts (within-CoT SD < 0.03, across-CoT SD > 0.43), then 10 rollouts at every sentence prefix of all 5 CoTs, for all eligible prompts | hours: ~2.4k rollouts per prompt, and the prompt count is unknown until Phase 3 |
| 5 | Score all rollouts (one judge server per GPU) and repair failed judge calls | ~1–3 h |

**Monitoring:**
- Progress lines go to stdout.
- Per-phase logs and run metadata go to `logs/r1_quadrants/<timestamp>_<job>/`: git SHA, model
  snapshots, vLLM and torch versions, and GPUs.
- Phase 4 logs one line per prompt, and `results/deepseek-ai/DeepSeek-R1/dataset/quadrants/run_metadata/*.json`
  lists the prompts selected.

## 3. What to send back

```bash
tar czf r1_results.tgz results/deepseek-ai/DeepSeek-R1/dataset logs/r1_quadrants
```

Send this via Hugging Face (a private dataset), scp, or anything else. Also send the exact vLLM
version and GPU type if they differ from `metadata.txt`. We will run the figures and the
commitment analysis from these files.

## Things to check

- **Template.** The `chat_template_renders` line in `logs/r1_quadrants/*/metadata.txt` should end in
  `<｜Assistant｜><think>\n`, which is the official R1 template (revision 56d4cbbb, the same
  convention as the distills). If it does not, stop and tell us.
- **No eligible prompts.** If Phase 4 reports `Sampled 0 from quadrant`, R1 has no quadrant prompts
  in the pool. Report this rather than loosening the thresholds; it is a result in itself.
- **Long CoTs.** R1 thinks for longer than the distills. In Phase 2, CoTs that hit 2,048 tokens
  without `</think>` are dropped (as for the other models). The baseline log reports how many.
