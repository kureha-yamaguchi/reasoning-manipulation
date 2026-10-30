#!/usr/bin/env bash
# Runs ON the RunPod pod (started by run_r1_on_runpod.sh): installs the repo at a pinned commit,
# downloads full-precision DeepSeek-R1 + the gpt-oss-20b judge to the pod volume, runs the
# quadrant pipeline, and tars the results.
#
# Cost safety, independent of the controlling machine:
#   - watchdog: stops the pod (volume kept) after CAP_SECONDS from boot
#   - on DONE, waits up to 2 h for /workspace/PULLED (results copied off) then stops the pod;
#     on FAILED, waits 30 min then stops. The controller normally stops it sooner via the API.
# Env: GIT_SHA (required), CAP_SECONDS (required); RunPod key in /workspace/.rp_key (mode 600).
set -euo pipefail
cd /workspace
: "${GIT_SHA:?}" "${CAP_SECONDS:?}"
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv-cache PATH="$HOME/.local/bin:$PATH"
log() { echo "[$(date '+%F %T')] $*"; }

stop_pod() {
    curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $(cat /workspace/.rp_key)" \
        https://api.runpod.io/graphql \
        -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}"
}
( sleep "$CAP_SECONDS"; log "Cost cap reached; stopping pod"; stop_pod ) >> /workspace/watchdog.log 2>&1 &

finish() {  # $1 = DONE | FAILED
    touch "/workspace/$1"; log "$1"
    local wait=$([ "$1" = DONE ] && echo 7200 || echo 1800)
    for _ in $(seq 1 $((wait / 30))); do [ -f /workspace/PULLED ] && break; sleep 30; done
    log "Stopping pod"; stop_pod
}
trap 'finish FAILED' ERR

log "Host: $(hostname); GPUs:"; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
apt-get update -qq && apt-get install -y -qq tmux rsync > /dev/null
curl -LsSf https://astral.sh/uv/0.5.21/install.sh | sh > /dev/null

if [ ! -d repo ]; then git clone -q https://github.com/kureha-yamaguchi/reasoning-manipulation.git repo; fi
cd repo && git fetch -q origin && git checkout -q "$GIT_SHA"
log "Repo at $(git rev-parse HEAD)"
uv sync -q
# Blackwell (compute capability >= 10, e.g. B200/B300): the locked torch 2.7.1 is the cu126 build,
# which has no sm_100 kernels. Swap in the same versions built for CUDA 12.8 (vLLM 0.10.1's own
# wheel already targets cu128/Blackwell) and stop `uv run` re-syncing back to the lock.
if [ "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1)" -ge 10 ]; then
    log "Blackwell GPU: installing torch 2.7.1+cu128"
    uv pip install -q --reinstall --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
    export UV_NO_SYNC=1
fi
log "torch $(uv run python -c 'import torch; print(torch.__version__, torch.cuda.get_device_capability())')"

log "Downloading weights"
uv run hf download deepseek-ai/DeepSeek-R1 --max-workers 32 > /workspace/download.log 2>&1
uv run hf download openai/gpt-oss-20b --max-workers 16 >> /workspace/download.log 2>&1
log "Weights: $(du -sh "$HF_HOME/hub" | cut -f1)"

log "Pipeline"
MODEL=deepseek-ai/DeepSeek-R1 TP=8 bash bash_scripts/r1_quadrants.sbatch
tar czf /workspace/r1_results.tgz results/deepseek-ai/DeepSeek-R1/dataset logs/r1_quadrants logs/score_quadrants
trap - ERR
finish DONE
