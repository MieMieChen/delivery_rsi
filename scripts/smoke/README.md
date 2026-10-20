# Check-in smoke test

A hand-written task that is not a delivery, run through the whole stack with the
trainer untouched: `TaskPlugin` → world with an event ledger → one generic
`GymImageEnv` → floor and ceiling references → GRPO.

| piece | file |
|---|---|
| task plugin: generator, `check_in` Tool, ledger reward, verifier | `embodiedbench/tasks/checkin.py` |
| world that executes the spec and writes the ledger | `embodiedbench/runtime/city/checkin_env.py` |
| generic VAGEN env (task registry) | `embodiedbench/training/vagen_task_env.py` |
| floor / ceiling on the same seeds | `tools/checkin_reference.py` |
| train / val episodes | `experiments/smoke_checkin_{train,val}.yaml` |
| CPU tests | `tests/test_checkin_task.py` |

The task: walk to two street addresses in Paris, in the listed order, and call
`check_in()` at each inside its time window. Reward is paid only from ledger
events (arrived 0.2, checked in on time 1.0, late 0.3, refused check-in −0.05).
Text-only, so no image albums are needed.

## Run (serv4)

```bash
git fetch origin && git checkout smoke/checkin-task
MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct bash scripts/smoke/run_checkin_smoke.sh
```

Defaults: GPUs `0,1,2,3` (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, so these are
nvidia-smi's indices; leave out a broken card), 20 GRPO steps of 4 prompts × 4
samples, validation at steps 0, 10 and 20 on 8 held-out seeds, no checkpoints.
`SKIP_GRPO=1` stops after the preflight. The interpreter is whatever `python3`
already has torch/vllm/ray/vagen, otherwise `$REPO/.venv-rl` is activated
(`VENV=` to point elsewhere).

Output goes to `exps/smoke_checkin_<UTC time>/`; send that directory back,
`SMOKE_SUMMARY.md` first. The run passes when the trainer exits 0, every step
logs a non-zero gradient on at least half the steps, rewards vary inside GRPO
groups, validation runs, the env logs finished episodes, and the ceiling solves
every validation seed. Whether the policy improves in 20 steps is reported, not
required.
