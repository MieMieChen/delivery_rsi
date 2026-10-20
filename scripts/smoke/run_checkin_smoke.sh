#!/usr/bin/env bash
# Smoke test: a hand-written, non-delivery task through the whole stack.
#
#   TaskPlugin (embodiedbench/tasks/checkin.py)  generates the episode, prices
#                                                ledger events, verifies
#   CheckInEnv (runtime/city/checkin_env.py)     executes it, writes the ledger
#   TaskGymEnv (training/vagen_task_env.py)      the one generic GymImageEnv
#   floor / ceiling (tools/checkin_reference.py) bracket it on the same seeds
#   GRPO (train_grpo_courier.sh -> VAGEN/verl)   trains on it, trainer untouched
#
# Usage, from anywhere:
#
#   MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct bash scripts/smoke/run_checkin_smoke.sh
#
# Everything lands in one directory ($OUT, default exps/smoke_checkin_<UTC time>),
# which is the thing to send back:
#
#   env_report.txt         versions, GPUs, git revision
#   stage2_tests.log       CPU tests of the task (no GPU)
#   references/            floor and ceiling on the validation seeds
#   stage4_preflight.log   VAGEN registry, yamls, prompt length vs the budgets
#   smoke_checkin.log      the trainer's own log, with the per-step metrics lines
#   episodes.jsonl         one line per finished episode: verdict and ledger
#   rollout_data/ validation_data/   the trainer's per-sample dumps
#   SMOKE_SUMMARY.md       pass/fail of every check, and the numbers
#
# Knobs (environment variables):
#   MODEL_PATH            required from stage 4 on
#   VENV                  virtualenv to activate if none is active (default $REPO/.venv-rl)
#   CUDA_VISIBLE_DEVICES  default 0,1,2,3; CUDA_DEVICE_ORDER is pinned to PCI_BUS_ID so
#                         the indices are nvidia-smi's (serv4's GPU 4 is broken: leave it out)
#   TOTAL_STEPS           GRPO steps (default 20); TEST_FREQ validates every N (default 10)
#   TRAIN_BS, GROUP       prompts per step and samples per prompt (default 4 x 4)
#   SKIP_GRPO=1           stop after the preflight (stages 1-4 need no GPU but stage 4's check)
#   OUT                   output directory
# Any arguments are passed to the trainer as extra Hydra overrides.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO" || exit 1
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${OUT:-$REPO/exps/smoke_checkin_$STAMP}
mkdir -p "$OUT"
# Absolute, because the trainer launcher cd's into vendor/vagen and the Ray
# workers have their own cwd: a relative path would land somewhere else.
OUT=$(cd "$OUT" && pwd)
# A reused directory must not lend this run its last run's results.
rm -f "$OUT/episodes.jsonl" "$OUT/metrics.jsonl" "$OUT/smoke_checkin.log" \
      "$OUT/SMOKE_SUMMARY.md" "$OUT/STATUS"
echo "smoke output: $OUT"

fail() { echo "SMOKE FAILED at $1 -- see $OUT"; echo "FAILED: $1" > "$OUT/STATUS"; exit 1; }

# ── environment ─────────────────────────────────────────────────────────────
VENV=${VENV:-$REPO/.venv-rl}
# Whatever python is first on PATH is used if it already has the training
# stack (an active venv or conda env); otherwise $VENV is activated. Checking
# the stack rather than VIRTUAL_ENV, because a lighter venv being active is
# exactly how the wrong interpreter gets picked.
if ! python3 -c "import torch, vllm, ray, vagen" >/dev/null 2>&1 \
        && [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi
export PYTHON=${PYTHON:-python3}
if [ -z "${CUDA_HOME:-}" ]; then
    for candidate in "$REPO/toolchains/cuda-12.8" "$(dirname "$(dirname "$(command -v "$PYTHON")")")"; do
        if [ -x "$candidate/bin/nvcc" ]; then export CUDA_HOME="$candidate"; export PATH="$CUDA_HOME/bin:$PATH"; break; fi
    done
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
export N_GPUS
export PYTHONPATH="$REPO:$REPO/vendor/vagen:${PYTHONPATH:-}"

# ── stage 1: what this machine is ───────────────────────────────────────────
{
    echo "date: $STAMP"
    echo "host: $(hostname)"
    echo "repo: $REPO"
    echo "git: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
    git status --short | head -20
    echo "python: $(command -v "$PYTHON") $("$PYTHON" --version 2>&1)"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER) N_GPUS=$N_GPUS"
    echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
    (command -v nvcc >/dev/null && nvcc --version | tail -2) || echo "nvcc: not on PATH"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>&1
    "$PYTHON" - <<'PYEOF' 2>&1
import importlib
for name in ("torch", "vllm", "ray", "transformers", "flash_attn", "verl", "vagen", "omegaconf", "numpy"):
    try:
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'ok')}")
    except Exception as error:  # noqa: BLE001
        print(f"{name}: MISSING ({type(error).__name__}: {error})")
try:
    import torch
    print("torch.cuda:", torch.version.cuda, "device_count:", torch.cuda.device_count())
except Exception as error:  # noqa: BLE001
    print("torch.cuda: unavailable", error)
PYEOF
} > "$OUT/env_report.txt" 2>&1
cat "$OUT/env_report.txt"

# ── stage 2: the task on CPU ────────────────────────────────────────────────
echo "== stage 2: CPU tests of the task"
"$PYTHON" -m pytest tests/test_checkin_task.py -q -p no:cacheprovider > "$OUT/stage2_tests.log" 2>&1 \
    || { tail -30 "$OUT/stage2_tests.log"; fail "stage 2 (CPU tests)"; }
tail -1 "$OUT/stage2_tests.log"

# ── stage 3: floor and ceiling on the validation seeds ──────────────────────
echo "== stage 3: floor and ceiling"
"$PYTHON" tools/checkin_reference.py --yaml experiments/smoke_checkin_val.yaml \
    --out-dir "$OUT/references" > "$OUT/stage3_references.log" 2>&1 \
    || { cat "$OUT/stage3_references.log"; fail "stage 3 (references)"; }
grep -v '^obs_size' "$OUT/stage3_references.log"

# ── GRPO sizing: 4 x 24 GB, text-only, as the measured prompt needs ─────────
export TRAIN_BS=${TRAIN_BS:-4} GROUP=${GROUP:-4} MICRO_BS=${MICRO_BS:-1}
export SP=${SP:-2} OFFLOAD=${OFFLOAD:-True}
# RESP_LEN holds every turn's observation as well as every reply; the notes
# grow, so a 20-turn episode reaches ~15k. Measured in the preflight.
export PROMPT_LEN=${PROMPT_LEN:-4608} RESP_LEN=${RESP_LEN:-16384}
export ROLLOUT_CTX=$((PROMPT_LEN + RESP_LEN))
export ROLLOUT_SEQS=${ROLLOUT_SEQS:-16} ROLLOUT_TOKENS=${ROLLOUT_TOKENS:-16384} ROLLOUT_MEM=${ROLLOUT_MEM:-0.65}
export TOTAL_STEPS=${TOTAL_STEPS:-20} TEST_FREQ=${TEST_FREQ:-10} SAVE_FREQ=${SAVE_FREQ:-0} TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
export ACTOR_LR=${ACTOR_LR:-1e-6} KL_COEF=${KL_COEF:-0.005} ENTROPY_COEF=${ENTROPY_COEF:-0.0} EAGER=${EAGER:-True}
export PROJECT_NAME=smoke_checkin EXPERIMENT_NAME=smoke_checkin EXPERIMENT_DIR="$OUT"
export DATASET_TRAIN=experiments/smoke_checkin_train.yaml DATASET_VAL=experiments/smoke_checkin_val.yaml
# Console only. verl's file logger json.dumps the metrics as they are, and
# num_turns/min and /max are numpy int32 -- it would kill the run at step 0.
# The console lines are tee'd into smoke_checkin.log and parsed from there.
export TRAIN_LOGGER="['console']"
export TASK_EPISODE_LOG="$OUT/episodes.jsonl"
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_DIR="$OUT"
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_${USER:-smoke}}
AGENT_WORKERS=${AGENT_WORKERS:-4}
BATCH=$((TRAIN_BS * GROUP))
if [ $((BATCH % N_GPUS)) -ne 0 ] || [ $((BATCH % AGENT_WORKERS)) -ne 0 ] || [ $((N_GPUS % SP)) -ne 0 ]; then
    fail "sizing: TRAIN_BS*GROUP=$BATCH must divide by N_GPUS=$N_GPUS and AGENT_WORKERS=$AGENT_WORKERS, and SP=$SP must divide N_GPUS"
fi

# ── stage 4: preflight (needs MODEL_PATH; opens every visible GPU) ──────────
echo "== stage 4: preflight"
: "${MODEL_PATH:?set MODEL_PATH to the Qwen3-VL-4B-Instruct directory}"
export MODEL_PATH="${MODEL_PATH%/}"
# Its trial episodes must not reach episodes.jsonl, which stage 6 reads as
# evidence that the TRAINER ran episodes.
TASK_EPISODE_LOG="" "$PYTHON" scripts/smoke/checkin_preflight.py > "$OUT/stage4_preflight.log" 2>&1 \
    || { cat "$OUT/stage4_preflight.log"; fail "stage 4 (preflight)"; }
grep -v '^obs_size' "$OUT/stage4_preflight.log"
if [ "${SKIP_GRPO:-0}" = "1" ]; then
    echo "SKIP_GRPO=1: stopping before training"; echo "PASSED: stages 1-4" > "$OUT/STATUS"; exit 0
fi

# ── stage 5: GRPO, the unmodified trainer ───────────────────────────────────
echo "== stage 5: GRPO ($TOTAL_STEPS steps, $TRAIN_BS x $GROUP per step, GPUs $CUDA_VISIBLE_DEVICES)"
NUM_CPUS=$(nproc); [ "$NUM_CPUS" -gt 32 ] && NUM_CPUS=32
bash embodiedbench/training/vagen/train_grpo_courier.sh \
    +env_registry.TaskEnv=embodiedbench.training.vagen_task_env.TaskGymEnv \
    actor_rollout_ref.rollout.agent.num_workers="$AGENT_WORKERS" \
    data.dataloader_num_workers=0 \
    trainer.resume_mode=disable \
    ray_kwargs.ray_init.num_cpus="$NUM_CPUS" \
    ++ray_kwargs.ray_init.num_gpus="$N_GPUS" \
    ++ray_kwargs.ray_init.address=local \
    ++ray_kwargs.ray_init.include_dashboard=False \
    ++ray_kwargs.ray_init.object_store_memory=2147483648 \
    "$@"
TRAIN_RC=$?

# ── stage 6: did it work ────────────────────────────────────────────────────
echo "== stage 6: summary"
"$PYTHON" scripts/smoke/summarize_checkin_smoke.py "$OUT" --trainer-rc "$TRAIN_RC" \
    --total-steps "$TOTAL_STEPS"
SUMMARY_RC=$?
if [ "$SUMMARY_RC" -eq 0 ]; then echo "PASSED" > "$OUT/STATUS"; else echo "FAILED: see SMOKE_SUMMARY.md" > "$OUT/STATUS"; fi
echo "send back: $OUT (SMOKE_SUMMARY.md first)"
exit "$SUMMARY_RC"
