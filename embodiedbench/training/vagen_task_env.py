"""One VAGEN ``GymImageEnv`` for any registered task, not just the courier's.

``CourierGymEnv`` hard-codes the delivery job. ``TaskGymEnv`` keeps everything
it does for a city episode -- the road network cache, the harness, the
observation and image plumbing -- and takes the three task-specific parts from
a registry entry instead:

* a **TaskPlugin** (``tasks/core.py``) that generates the frozen
  ``EpisodeSpec``, prices ledger events and verifies the episode;
* a **world** class with ``from_episode(network, spec, **kw)``, an append-only
  ``ledger``, ``poll_events()`` and ``privileged_state()``;
* a **session** class, the harness that renders that world for the policy.

Reward is the sum of the plugin's ``event_reward`` over the ledger events each
step added -- nothing the policy says is ever paid. ``info["success"]`` is the
plugin's verifier, which is also what ends a finished episode early in the
agent loop. The trainer (``vendor/``) is untouched: the env is registered from
the launch command line as ``+env_registry.TaskEnv=<this module>.TaskGymEnv``.

Config keys, besides every ``CourierGymEnv`` key:

==================  ==========================================================
``task``            registry name (default ``checkin``)
``task_config``     mapping passed to the plugin (e.g. ``stops``, ``max_leg_cm``)
``progress_weight`` potential-based shaping toward the current target, off by
                    default; training-time only, never in the verdict
==================  ==========================================================

``TASK_EPISODE_LOG`` (env var) or ``task_episode_log`` (config) names a JSONL
file that gets one line per finished episode: seed, instance, verdict, ledger
counts. Ray workers inherit the variable from the launcher.
"""

from __future__ import annotations

import importlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any

from embodiedbench.training.vagen_courier_env import PROGRESS_SCALE_CM, CourierGymEnv


@dataclass(frozen=True)
class TaskEntry:
    """Where a task's three parts live, as ``module:attribute`` strings."""

    plugin: str
    world: str
    session: str


TASKS: dict[str, TaskEntry] = {
    "checkin": TaskEntry(
        plugin="embodiedbench.tasks.checkin:CheckInTask",
        world="embodiedbench.runtime.city.checkin_env:CheckInEnv",
        session="embodiedbench.runtime.city.checkin_env:CheckInSession",
    ),
}


def _load(path: str) -> Any:
    module, _, attribute = path.partition(":")
    return getattr(importlib.import_module(module), attribute)


#: Road networks by map directory, per process. The agent loop builds a new env
#: object for every trajectory, and building the network was ~0.2 s of each
#: reset -- run synchronously on the worker's shared event loop. The network is
#: read-only once built, so one copy per process is safe.
_NETWORKS: dict[str, Any] = {}


class TaskGymEnv(CourierGymEnv):
    """A city episode of whichever registered task the config names."""

    def __init__(self, env_config: dict[str, Any] | None = None):
        config = dict(env_config or {})
        self.task_name = str(config.get("task") or "checkin")
        if self.task_name not in TASKS:
            raise ValueError(f"unknown task {self.task_name!r}; registered: {sorted(TASKS)}")
        self.task_config = dict(config.get("task_config") or {})
        super().__init__(config)
        entry = TASKS[self.task_name]
        self.task = _load(entry.plugin)(self.task_config)
        self._world_cls = _load(entry.world)
        self._session_cls = _load(entry.session)
        self.task_episode_log = (config.get("task_episode_log")
                                 or os.environ.get("TASK_EPISODE_LOG") or "")
        self._spec: Any = None
        self._ledger_seen = 0
        self._logged = False

    # ── the CourierGymEnv hooks ──────────────────────────────────────────────

    def _make_world(self, default_cls: type, kwargs: dict[str, Any]) -> Any:
        seed = int(kwargs["seed"])
        spec = self.task.generate(self._network, seed)
        self.task.check_solvable(spec, self._network).require()
        self._spec = spec
        world_kwargs = {**kwargs, **getattr(self.task, "world_kwargs", dict)()}
        return self._world_cls.from_episode(self._network, spec, **world_kwargs)

    def _make_session(self, default_cls: type, world: Any) -> Any:
        return self._session_cls(world, city=self.city)

    # ── VAGEN interface ──────────────────────────────────────────────────────

    async def reset(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._network is None and not self.map_rotation and self.album_root is None:
            from embodiedbench.compiler.road_network import build_road_network

            key = str(self.map_dir.resolve())
            if key not in _NETWORKS:
                _NETWORKS[key] = build_road_network(self.map_dir, map_name=self.map_dir.name)
            self._network = _NETWORKS[key]
        obs, info = await super().reset(seed)
        # Anything the world wrote while setting up is not the policy's doing.
        self._ledger_seen = len(self._env.ledger)
        self._logged = False
        info.update({"task": self.task_name, "instance_id": self._spec.instance_id})
        return obs, info

    async def step(self, action_str: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._session is None:
            raise RuntimeError("reset() before step()")
        before_cm, before_target = self._remaining_cm()
        log = self._session.step(action_str)
        self._turns += 1
        self._env.poll_events()

        new_events = self._env.ledger[self._ledger_seen:]
        self._ledger_seen = len(self._env.ledger)
        reward = float(sum(self.task.event_reward(event) for event in new_events))

        # Potential-based shaping, as in CourierGymEnv: per turn, and skipped on
        # the turn the target changes (a check-in moves it to the next stop).
        after_cm, after_target = self._remaining_cm()
        step_progress = 0.0
        if (self.progress_weight and before_cm is not None and after_cm is not None
                and before_target == after_target):
            step_progress = (before_cm - after_cm) / PROGRESS_SCALE_CM
            self._progress_cm += before_cm - after_cm
            reward += self.progress_weight * step_progress

        done = bool(self._session.finished)
        if self.max_turns and self._turns >= self.max_turns:
            done = True
        verdict = self.task.verify(self._env.privileged_state())
        obs = {"obs_str": ""} if done else self._observation()[0]
        if done:
            self._log_task_episode(verdict)
        info = {
            "status": log.status,
            "action": log.action,
            "error": log.error,
            "sim_seconds": log.sim_seconds,
            "turns": self._turns,
            "task": self.task_name,
            # The verifier's verdict. It is also the agent loop's early stop,
            # and it can only be true once every stop is checked in.
            "success": bool(verdict["success"]),
            "checked_in": int(verdict["checked_in"]),
            "on_time": int(verdict["on_time"]),
            "in_order": bool(verdict["in_order"]),
            "ledger_reward": float(verdict["ledger_reward"]),
            "ledger_events": len(new_events),
            "step_progress": round(step_progress, 4),
            # The agent loop logs these courier columns for every env; filled
            # with the task's nearest equivalents so the curves mean something.
            "delivered": int(verdict["checked_in"]),
            "earnings": float(verdict["on_time"]),
            "termination": self._session.run.termination_reason,
        }
        return obs, reward, done, info

    async def close(self) -> None:
        # The agent loop can end an episode without the env ever saying done --
        # on its response-token budget -- and it always closes the env after.
        # Such an episode still gets its line, marked as cut.
        if self._session is not None and self._env is not None and not self._logged:
            self._log_task_episode(self.task.verify(self._env.privileged_state()),
                                   termination="agent_loop_cut")
        await super().close()

    def _log_task_episode(self, verdict: dict[str, Any], termination: str | None = None) -> None:
        self._logged = True
        if not self.task_episode_log:
            return
        try:
            row = {
                "task": self.task_name,
                "seed": int(getattr(self, "_episode_seed", -1)),
                "instance_id": self._spec.instance_id if self._spec else None,
                "turns": self._turns,
                "termination": termination or self._session.run.termination_reason,
                "events": dict(Counter(e["kind"] for e in self._env.ledger)),
                **{k: verdict[k] for k in ("success", "stops", "checked_in", "on_time",
                                           "in_order", "ledger_reward", "failure")},
            }
            line = (json.dumps(row) + "\n").encode()
            fd = os.open(self.task_episode_log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        except OSError:
            pass  # telemetry must never take a rollout down
