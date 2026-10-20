#!/usr/bin/env python3
"""Stage 6 of scripts/smoke/run_checkin_smoke.sh: decide whether the smoke passed.

    python scripts/smoke/summarize_checkin_smoke.py exps/smoke_checkin_<time> [--trainer-rc 0]

Reads the run directory -- the trainer's log (its ``step:N - key:value`` metric
lines; metrics.jsonl from verl's file logger is read instead if present),
episodes.jsonl (one line per finished episode, written by TaskGymEnv) and the
floor/ceiling references -- and writes SMOKE_SUMMARY.md. Exits 0 only if every
check that decides "the plumbing works" passed. Whether the policy *learned*
anything in a few dozen steps is reported, not required: a smoke test is about
the pipe, and twenty steps of a 4B are not evidence either way.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

VAL_SOURCE = "checkin_val"
STEP_LINE = re.compile(r"step:(\d+) - (.*)")
#: numpy >= 2 prints scalars as ``np.float64(0.3)``; unwrap before float().
NUMPY_REPR = re.compile(r"^np\.\w+\((.*)\)$")

#: Exact keys the trainer emits (verl metric_utils, dp_actor; vagen ray_trainer).
GRAD_NORM = "actor/grad_norm"
PG_LOSS = "actor/pg_loss"
SCORE = "critic/score/mean"
REWARD_VARIANCE = "custom_metrics/train/reward_variance"
TRAIN_TURNS = "num_turns/mean"
RESPONSE_LENGTH = "response_length/mean"
VAL_REWARD = f"val-core/{VAL_SOURCE}/reward/mean@1"
VAL_SUCCESS = f"val-aux/{VAL_SOURCE}/traj_success/mean@1"
VAL_TURNS = "val-aux/num_turns/mean"


def load_metrics(out: Path) -> dict[int, dict[str, float]]:
    """Per-step metrics, from the file logger, else from the console log."""
    steps: dict[int, dict[str, float]] = {}
    path = out / "metrics.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            steps.setdefault(int(row["step"]), {}).update(
                {k: v for k, v in row.get("data", {}).items() if isinstance(v, (int, float))})
    if steps:
        return steps
    for log in out.glob("*.log"):
        for line in log.read_text(errors="replace").splitlines():
            match = STEP_LINE.search(line)
            if not match:
                continue
            values = steps.setdefault(int(match.group(1)), {})
            for item in match.group(2).split(" - "):
                key, _, value = item.partition(":")
                value = NUMPY_REPR.sub(r"\1", value.strip())
                try:
                    values[key.strip()] = float(value)
                except ValueError:
                    pass
    return steps




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--trainer-rc", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=0)
    args = parser.parse_args(argv)
    out = args.out

    log_text = "\n".join(p.read_text(errors="replace") for p in sorted(out.glob("smoke_checkin*.log")))
    metrics = load_metrics(out)
    train_steps = sorted(s for s, m in metrics.items() if GRAD_NORM in m)
    val_steps = sorted(s for s, m in metrics.items() if VAL_REWARD in m)
    episodes = [json.loads(l) for l in (out / "episodes.jsonl").read_text().splitlines()
                if l.strip()] if (out / "episodes.jsonl").exists() else []
    train_eps = [e for e in episodes if e["seed"] < 1000]
    val_eps = [e for e in episodes if e["seed"] >= 1000]
    refs = {name: json.loads((out / "references" / f"{name}.json").read_text())
            for name in ("floor", "ceiling") if (out / "references" / f"{name}.json").exists()}

    grad = [metrics[s].get(GRAD_NORM) or 0.0 for s in train_steps]
    variance = [metrics[s].get(REWARD_VARIANCE) for s in train_steps]
    variance = [v for v in variance if v is not None]
    step_failures = log_text.count("Environment step failed")
    tracebacks = len(re.findall(r"^Traceback", log_text, re.M))

    # (name, passed, detail, required). Required checks decide the verdict; the
    # rest are reported because they are worth reading, not because a smoke
    # run can be failed on them (vLLM and Ray print benign tracebacks).
    checks: list[tuple[str, bool, str, bool]] = [
        ("trainer exited 0", args.trainer_rc == 0, f"rc {args.trainer_rc}", True),
        ("all training steps ran",
         bool(train_steps) and (not args.total_steps or max(train_steps) >= args.total_steps),
         f"{len(train_steps)} steps logged, last {max(train_steps) if train_steps else None}", True),
        ("no env step raised", step_failures == 0, f"{step_failures} 'Environment step failed'", True),
        ("no traceback in the trainer log", tracebacks == 0, f"{tracebacks} tracebacks", False),
        ("gradients flowed", sum(g > 0 for g in grad) >= max(1, len(grad) // 2),
         f"grad_norm > 0 on {sum(g > 0 for g in grad)}/{len(grad)} steps", True),
        ("rewards varied inside groups", any(v > 0 for v in variance),
         f"reward_variance > 0 on {sum(v > 0 for v in variance)}/{len(variance)} steps", True),
        ("validation ran before and during training", len(val_steps) >= 2,
         f"validation at steps {val_steps}", True),
        ("the env logged finished episodes", bool(train_eps) and bool(val_eps),
         f"{len(train_eps)} training, {len(val_eps)} validation episodes", True),
        ("the references bracket the task",
         refs.get("ceiling", {}).get("success_rate") == 1.0,
         f"ceiling success {refs.get('ceiling', {}).get('success_rate')}", True),
    ]
    passed = all(result for _, result, _, required in checks if required)

    def val_at(step: int, key: str) -> str:
        value = metrics.get(step, {}).get(key)
        return "—" if value is None else f"{value:.3f}"

    lines = [f"# Check-in smoke: {'PASSED' if passed else 'FAILED'}", "",
             f"Run directory: `{out}`", "", "## Checks", "",
             "| check | result | detail |", "|---|---|---|"]
    lines += [f"| {name}{'' if required else ' (reported only)'} | "
              f"{'pass' if result else ('**FAIL**' if required else 'no')} | {detail} |"
              for name, result, detail, required in checks]
    lines += ["", "## Validation reward (ledger reward, 8 held-out seeds)", "",
              "| | reward | success | turns |", "|---|---|---|---|"]
    for name in ("floor", "ceiling"):
        r = refs.get(name)
        if r:
            lines.append(f"| {name} reference | {r['ledger_reward_mean']:.3f} | "
                         f"{r['success_rate']:.3f} | {r['turns_mean']:.1f} |")
    for step in val_steps:
        lines.append(f"| policy, step {step} | {val_at(step, VAL_REWARD)} | "
                     f"{val_at(step, VAL_SUCCESS)} | {val_at(step, VAL_TURNS)} |")
    lines += ["", "## Training steps", "",
              "| step | score mean | reward variance | grad_norm | pg_loss | turns | response length |",
              "|---|---|---|---|---|---|---|"]
    for step in train_steps:
        m = metrics[step]

        def fmt(key: str) -> str:
            value = m.get(key)
            return "—" if value is None else f"{value:.4g}"
        lines.append(f"| {step} | {fmt(SCORE)} | {fmt(REWARD_VARIANCE)} | "
                     f"{fmt(GRAD_NORM)} | {fmt(PG_LOSS)} | "
                     f"{fmt(TRAIN_TURNS)} | {fmt(RESPONSE_LENGTH)} |")
    if episodes:
        def rate(rows: list[dict[str, Any]], key: str) -> str:
            return f"{sum(bool(r[key]) for r in rows) / len(rows):.3f}" if rows else "—"
        lines += ["", "## Episodes (TaskGymEnv's log)", "",
                  "| split | episodes | success | mean check-ins | mean ledger reward |",
                  "|---|---|---|---|---|"]
        for name, rows in (("train", train_eps), ("validation", val_eps)):
            if rows:
                lines.append(f"| {name} | {len(rows)} | {rate(rows, 'success')} | "
                             f"{sum(r['checked_in'] for r in rows) / len(rows):.2f} | "
                             f"{sum(r['ledger_reward'] for r in rows) / len(rows):.3f} |")
    (out / "SMOKE_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
