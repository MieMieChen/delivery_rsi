#!/usr/bin/env python3
"""Floor and ceiling for the ``checkin`` task, on the same episodes GRPO sees.

    python tools/checkin_reference.py --yaml experiments/smoke_checkin_val.yaml --out-dir results/checkin

Reads the task block of a VAGEN env yaml (``task_config``, ``stride``,
``max_turns`` and the seed range), so the references are scored on exactly the
episodes and turn budget the trainer uses. For every seed it generates the
``EpisodeSpec`` with the task plugin, checks it is statically solvable, builds
the world from the spec, and runs:

* the **floor**, ``ObservationOnlyCourier``: reads only what the world says;
* the **ceiling**, the better per seed of ``FewestMovesCourier`` and
  ``ShortestPathCourier``: reads the road graph.

Each run is judged by the task's own verifier and paid by its own ledger
reward, then written in ``waypoint_reference.summarise``'s shape (``earnings``
there is the on-time check-in count) plus the verifier's numbers. Exits
non-zero if the ceiling fails to solve any seed: a task its privileged
reference cannot finish is not admitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from embodiedbench.compiler.road_network import build_road_network  # noqa: E402
from embodiedbench.runtime.city.checkin_env import CheckInEnv  # noqa: E402
from embodiedbench.tasks.checkin import CheckInTask  # noqa: E402
from tools import waypoint_reference as reference  # noqa: E402

DEFAULT_MAP = REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"


def task_block(path: Path) -> dict[str, Any]:
    """The first ``envs`` block of a VAGEN env yaml."""
    import yaml

    data = yaml.safe_load(path.read_text())
    block = data["envs"][0]
    if block.get("config", {}).get("task", "checkin") != "checkin":
        raise SystemExit(f"{path}: not a checkin block")
    return block


def seeds_of(block: dict[str, Any]) -> list[int]:
    seed = block["seed"]
    if isinstance(seed, list) and len(seed) == 3:
        first, last, stride = seed
        return list(range(int(first), int(last) + 1, int(stride)))
    if isinstance(seed, list):
        return [int(s) for s in seed]
    return [int(seed)]


def run_one(network: Any, task: CheckInTask, seed: int, cap: int, stride: str,
            objective: str) -> dict[str, Any]:
    spec = task.generate(network, seed)
    task.check_solvable(spec, network).require()
    world = CheckInEnv.from_episode(network, spec, stride=stride, **task.world_kwargs())
    world.reset()
    row = reference.run_shift(world, seed, cap, objective=objective)
    world.poll_events()
    verdict = task.verify(world.privileged_state())
    row.update({
        "objective": objective,
        "instance_id": spec.instance_id,
        "spec_sha256": spec.content_hash(),
        "success": verdict["success"],
        "checked_in": verdict["checked_in"],
        "ledger_reward": verdict["ledger_reward"],
        "failure": verdict["failure"],
        "events": [e["kind"] for e in world.ledger],
    })
    return row


def summary(rows: list[dict[str, Any]], *, objective: str, block: dict[str, Any],
            cap: int) -> dict[str, Any]:
    out = reference.summarise(rows, task={"task": "checkin", **block["config"]},
                              split=str(block.get("data_source", "checkin")), cap=cap,
                              objective=objective)
    n = max(1, len(rows))
    out.update({
        "success_rate": round(sum(r["success"] for r in rows) / n, 4),
        "checked_in_mean": round(sum(r["checked_in"] for r in rows) / n, 4),
        "ledger_reward_mean": round(sum(r["ledger_reward"] for r in rows) / n, 4),
        "turns_mean": round(sum(r["turns"] for r in rows) / n, 2),
    })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yaml", type=Path, default=REPO / "experiments/smoke_checkin_val.yaml")
    parser.add_argument("--out-dir", type=Path, default=REPO / "results/checkin")
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--max-seeds", type=int, default=0, help="0 = every seed in the yaml")
    args = parser.parse_args(argv)

    block = task_block(args.yaml)
    config = block["config"]
    cap = int(block.get("max_turns") or config.get("max_turns") or 60)
    stride = str(config.get("stride", "block"))
    seeds = seeds_of(block)[: args.max_seeds or None]
    task = CheckInTask(config.get("task_config") or {})
    network = build_road_network(args.map_dir, map_name=args.map_dir.name)

    floor = [run_one(network, task, s, cap, stride, "floor") for s in seeds]
    ceiling = []
    for seed in seeds:
        runs = [run_one(network, task, seed, cap, stride, o) for o in ("moves", "metres")]
        ceiling.append(max(runs, key=lambda r: (r["success"], r["ledger_reward"], -r["turns"])))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {"floor": summary(floor, objective="floor", block=block, cap=cap),
               "ceiling": summary(ceiling, objective="best", block=block, cap=cap)}
    for name, rows in (("floor", floor), ("ceiling", ceiling)):
        (args.out_dir / f"{name}.json").write_text(json.dumps(results[name], indent=2))
        with (args.out_dir / f"{name}.episodes.jsonl").open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    print(f"checkin references on {len(seeds)} seeds of {args.yaml.name} "
          f"(cap {cap} turns, stride {stride}, task_config {config.get('task_config')}):")
    for name in ("floor", "ceiling"):
        r = results[name]
        print(f"  {name:8s} success {r['success_rate']:.2f}  check-ins {r['checked_in_mean']:.2f}  "
              f"ledger reward {r['ledger_reward_mean']:.3f}  turns {r['turns_mean']:.1f}")
    unsolved = [r["seed"] for r in ceiling if not r["success"]]
    if unsolved:
        print(f"CEILING FAILED on seeds {unsolved}: the task is not admitted", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
