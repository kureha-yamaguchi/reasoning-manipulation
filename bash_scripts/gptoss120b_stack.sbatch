#!/bin/bash
#SBATCH --qos=math-lasr
#SBATCH --account=math-lasr
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=250G
#SBATCH --time=72:00:00
#SBATCH --job-name=gptoss120b_stack
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --mail-user=benjamin.etheridge@eng.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

# gpt-oss-120b on 2xH100 (MXFP4, ~65 GB: one copy fits per 94 GB GPU): the paper stack except Section 5
# and the capability evals (stages 2-3 need ~240 GB bf16 ablated copies, i.e. a 4xH100 node).
#   1  baseline on all 2,446 prompts, quadrants, 10-rollout prefix resampling (Fig 3)
#   4  Thought Anchors (15 seeded eligible prompts x 2 CoTs, 100 rollouts per sentence, stance, analyses)
#   5  prefill intervention
# The pipeline runs in the background; whether it finishes or fails, the job then holds the node until
# walltime so it can be inspected and resumed from a login shell, e.g.
#   srun --jobid=<id> --overlap --pty bash      (then START_STAGE=n ... bash bash_scripts/full_stack.sh)
# Submit from the repo root: mkdir -p logs/slurm && sbatch bash_scripts/gptoss120b_stack.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
MODEL=openai/gpt-oss-120b
TS=$(date +%Y%m%d_%H%M%S)
DRV=logs/full_stack_gptoss120b_driver.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$DRV"; }
log "Job $SLURM_JOB_ID on $(hostname), GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd,)"
nvidia-smi

(
    set -e
    echo "[$(date "+%F %T")] Downloading $MODEL"
    uv run hf download "$MODEL" --max-workers 16 > "logs/gptoss120b_download_$TS.log" 2>&1
    MODEL=$MODEL SKIP_STAGES="2 3" bash bash_scripts/full_stack.sh
) >> "$DRV" 2>&1 && log "Pipeline finished" || log "Pipeline FAILED (rc=$?); node kept for inspection"

log "Holding the node until walltime"
sleep infinity
