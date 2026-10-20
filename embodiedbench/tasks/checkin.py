"""``checkin@1``: walk to a list of street addresses, in order, and check in at each.

The first task in this repository that is not a delivery. It exists to prove the
plumbing a generated task will need, end to end, with a task a person wrote: a
``TaskPlugin`` generates a frozen ``EpisodeSpec``, a world executes it, the
world's event ledger pays the reward, the task's verifier judges the episode on
privileged state, the floor and ceiling couriers bracket it, and the unmodified
GRPO stack trains on it. Nothing here is specific to check-ins except the
constants; the shape is the one an agent-written task will have to fill.

What the task is made of, in the proposal's terms (local/proposal):

* **TaskSpec** -- ``generate`` returns an ``EpisodeSpec`` whose order schedule
  is the list of stops, in order, as StableId building ids with the address
  text in the payload. The spawn is pinned too, so the spec alone describes the
  episode played; the world must not re-draw either.
* **Task-specific action** -- ``CHECK_IN``, a Tool declaration: precondition
  (standing at the current stop's door), effect (a ``checked_in`` ledger event),
  cost (30 s and a turn), refusal (a turn and 5 s, like every refusal here, so
  it is never a free range-finder).
* **Reward** -- ``event_reward`` is a pure function of ledger events. The policy
  is never paid for anything it says, only for what the world recorded.
* **Verifier** -- ``verify`` reads privileged state: every stop checked in, in
  the listed order, each inside its window, none given up on.

The plugin imports no simulator code (``tasks/core.py``): the world that
executes the spec is ``embodiedbench.runtime.city.checkin_env``, and the glue
that pairs them for training is ``embodiedbench.training.vagen_task_env``.
"""

from __future__ import annotations

import hashlib
import heapq
import inspect
import json
import math
import random
from dataclasses import asdict, dataclass, fields
from typing import Any, Iterable, Mapping

from embodiedbench.agent.courier.tools import Tool, ToolKind
from embodiedbench.schemas.environment import NavigationMode
from embodiedbench.schemas.episode import Budgets, EpisodeSpec, ScheduledOrder, TaskConfigRef
from embodiedbench.schemas.geometry import FrameName, Pose, Vec3
from embodiedbench.schemas.runtime import RuntimeMode
from embodiedbench.schemas.trajectory import MetricValue, ScoreReport, Trajectory
from embodiedbench.tasks.core import SolvabilityVerdict, TaskPlugin, TaskRequirements

TASK_ID = "checkin"
TASK_VERSION = "0.1.0"
EVALUATOR_ID = "checkin_verifier"

#: Ledger event kinds this task's world writes and its reward reads.
EVENT_ARRIVED = "arrived"
EVENT_CHECKED_IN = "checked_in"
EVENT_REFUSED = "check_in_refused"

#: The whole reward, as a table over ledger events. Arrival is worth a little so
#: that a group of samples which all fail to check in still differ in how far
#: they got (GRPO needs variance inside a group); it is paid once per stop, so it
#: cannot be farmed by walking back and forth. A late check-in still counts as a
#: visit but is worth much less than an on-time one. A refused check-in costs a
#: little, so spamming the tool to find the door is never free.
REWARDS = {
    EVENT_ARRIVED: 0.2,
    EVENT_CHECKED_IN: 1.0,
    "checked_in_late": 0.3,
    EVENT_REFUSED: -0.05,
}

#: At least this many turns a stop. ``profiles.MIN_STEPS_PER_ORDER`` is 12 for a
#: delivery's two stops; one stop and no collection needs fewer.
MIN_STEPS_PER_STOP = 6

CHECK_IN = Tool(
    name="check_in",
    kind=ToolKind.ACT,
    summary=("Check in at the stop you are standing at. Only works at the "
             "current stop on your list, and only in the listed order."),
    example="check_in()",
    time_cost_s=30.0,
    returns="the stop is done and the next one on the list becomes current",
    refusals=(
        ("you are not at the current stop",
         "it costs a turn and five seconds and points nowhere; compare the "
         "street and door number at the top of the turn against your list "
         "instead of probing"),
    ),
    use_when="street and door number both match the current stop on your list",
    not_for="checking whether you have arrived",
)


@dataclass(frozen=True)
class CheckInConfig:
    """Everything that changes an episode, hashed into the spec."""

    stops: int = 3
    #: Bounds on each leg's shortest walk, spawn to stop 1 and stop to stop.
    min_leg_cm: float = 3000.0
    max_leg_cm: float = 15000.0
    #: Each stop's window is this times its leg's walking time, plus handling.
    deadline_slack: float = 2.5
    step_budget: int = 40
    #: Consecutive stops must be on different streets, so a leg is a choice of
    #: turns and not a walk along one street.
    distinct_streets: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "CheckInConfig":
        values = dict(values or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown checkin config keys {unknown}; known: {sorted(known)}")
        return cls(**values)

    def sha256(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


def _config(config: Any) -> CheckInConfig:
    if config is None:
        return CheckInConfig()
    if isinstance(config, CheckInConfig):
        return config
    return CheckInConfig.from_mapping(config)


def _graph_distances(nodes: Mapping[str, Any], start: str) -> dict[str, float]:
    """Shortest walking distance from ``start`` to every reachable node.

    The plugin's own Dijkstra over the compiled graph, so generation needs no
    simulator. Sorted neighbours keep ties, and therefore the spec, stable.
    """
    best = {start: 0.0}
    queue = [(0.0, start)]
    done: set[str] = set()
    while queue:
        cost, node = heapq.heappop(queue)
        if node in done:
            continue
        done.add(node)
        here = nodes[node].position
        for neighbour in sorted(nodes[node].neighbours):
            step = cost + math.dist(here, nodes[neighbour].position)
            if step < best.get(neighbour, math.inf):
                best[neighbour] = step
                heapq.heappush(queue, (step, neighbour))
    return best


class CheckInTask:
    """The ``checkin`` TaskPlugin: generator, action declaration, reward, verifier."""

    id = TASK_ID
    version = TASK_VERSION

    def __init__(self, config: CheckInConfig | Mapping[str, Any] | None = None):
        self.config = _config(config)

    # ── TaskPlugin protocol (tasks/core.py) ─────────────────────────────────

    def requirements(self, config: Mapping[str, Any] | None = None) -> TaskRequirements:
        cfg = self.config if config is None else _config(config)
        return TaskRequirements(affordances={"address": cfg.stops + 1})

    def generate(self, world: Any, seed: int,
                 config: Mapping[str, Any] | None = None) -> EpisodeSpec:
        """A frozen episode: spawn and stops, chosen here and nowhere else.

        ``world`` is the compiled ``RoadNetwork`` (nodes, addresses, map name).
        Inputs are sorted and the RNG is seeded from ``seed`` alone, so the
        same (map, config, seed) gives a byte-identical spec in any process.
        """
        cfg = self.config if config is None else _config(config)
        rng = random.Random(f"{TASK_ID}:{seed}")
        nodes = world.nodes
        junctions = sorted(n for n, node in nodes.items() if len(node.neighbours) >= 2)
        addresses = sorted(
            (a for a in world.addresses if a.kerb_node in nodes),
            key=lambda a: a.building_id)
        if len(addresses) < cfg.stops + 1 or not junctions:
            raise ValueError(f"{world.map_name}: not enough addresses for {cfg.stops} stops")

        for _ in range(200):
            spawn = rng.choice(junctions)
            stops = self._draw_stops(nodes, addresses, spawn, cfg, rng)
            if stops is not None:
                break
        else:
            raise ValueError(f"{world.map_name} seed {seed}: no stop chain fits the leg bounds")

        position = nodes[spawn].position
        schedule = [
            ScheduledOrder(
                order_id=f"stop_{i}",
                available_from_sim_time_s=0.0,
                pickup_site=address.building_id,
                dropoff_site=address.building_id,
                payload={"index": i, "address": address.text,
                         "kerb_node": address.kerb_node},
            )
            for i, address in enumerate(stops, start=1)
        ]
        return EpisodeSpec(
            instance_id=f"{world.map_name}_{TASK_ID}_{seed:06d}",
            environment_id=world.map_name,
            environment_version="0.1.0",
            task=TaskConfigRef(plugin=self.id, version=self.version,
                               config_sha256=cfg.sha256()),
            embodiment_profile="human_on_foot",
            navigation_mode=NavigationMode.NAV_WAYPOINT,
            runtime_track=RuntimeMode.TEXT,
            seed=int(seed),
            spawn=Pose(frame=FrameName.BUNDLE_WORLD,
                       position=Vec3(x_cm=position[0], y_cm=position[1])),
            order_schedule=schedule,
            observable_instruction=self.instruction(schedule),
            budgets=Budgets(steps=cfg.step_budget),
            evaluator_id=EVALUATOR_ID,
            evaluator_version=self.version,
            private_state_ref=f"private://{world.map_name}/{TASK_ID}/{seed}",
        )

    def world_kwargs(self) -> dict[str, Any]:
        """Constructor arguments the executing world needs from this config."""
        return {"deadline_slack": self.config.deadline_slack}

    def action_schema(self, state: Any = None) -> list[str]:
        """The actions this task adds on top of the world's atomic ones."""
        return [CHECK_IN.name]

    def evaluate(self, trajectory: Trajectory, privileged: dict[str, Any]) -> ScoreReport:
        """The verifier's verdict as a ScoreReport, for a recorded trajectory."""
        verdict = self.verify(privileged)
        invalid = sum(1 for turn in trajectory.turns
                      if turn.action_result.status.value != "accepted")
        metrics = {
            "stops": MetricValue(value=float(verdict["stops"])),
            "checked_in": MetricValue(value=float(verdict["checked_in"])),
            "on_time": MetricValue(value=float(verdict["on_time"])),
            "in_order": MetricValue(value=float(verdict["in_order"])),
            "ledger_reward": MetricValue(value=float(verdict["ledger_reward"])),
            "steps_taken": MetricValue(value=float(len(trajectory.turns))),
            "invalid_actions": MetricValue(value=float(invalid)),
        }
        return ScoreReport(
            instance_id=trajectory.instance_id,
            episode_id=trajectory.episode_id,
            evaluator_id=EVALUATOR_ID,
            evaluator_version=self.version,
            success=bool(verdict["success"]),
            normalized_utility_vs_upper_bound=None,
            upper_bound_undefined_reason=(
                "the ceiling courier is run separately (tools/checkin_reference.py); "
                "a per-episode ratio to it is not part of this verdict"),
            metrics=metrics,
            failure_reason=verdict["failure"],
            trajectory_sha256=trajectory.content_hash(),
        )

    # ── the parts an agent-written task would also have to supply ───────────

    def check_solvable(self, spec: EpisodeSpec, world: Any) -> SolvabilityVerdict:
        """Static feasibility: every stop exists and is reachable from the last.

        The empirical half -- the ceiling courier actually finishing -- lives in
        the tests and in ``tools/checkin_reference.py``.
        """
        reasons: list[str] = []
        nodes = world.nodes
        sites = {a.building_id: a for a in world.addresses}
        cursor = min(nodes, key=lambda n: math.dist(
            nodes[n].position, (spec.spawn.position.x_cm, spec.spawn.position.y_cm)))
        if math.dist(nodes[cursor].position,
                     (spec.spawn.position.x_cm, spec.spawn.position.y_cm)) > 1.0:
            reasons.append("spawn is not on a graph node")
        for order in spec.order_schedule:
            address = sites.get(order.pickup_site)
            if address is None or address.kerb_node not in nodes:
                reasons.append(f"{order.order_id}: site {order.pickup_site} has no kerb node")
                continue
            if address.kerb_node not in _graph_distances(nodes, cursor):
                reasons.append(f"{order.order_id}: unreachable from the previous stop")
            cursor = address.kerb_node
        if spec.budgets.steps < MIN_STEPS_PER_STOP * len(spec.order_schedule):
            reasons.append(f"step budget {spec.budgets.steps} < "
                           f"{MIN_STEPS_PER_STOP} x {len(spec.order_schedule)} stops")
        if not spec.order_schedule:
            reasons.append("no stops")
        return SolvabilityVerdict(feasible=not reasons, reasons=reasons)

    @staticmethod
    def event_reward(event: Mapping[str, Any]) -> float:
        """The reward one ledger event is worth. The only source of reward."""
        kind = event.get("kind")
        if kind == EVENT_CHECKED_IN:
            on_time = bool((event.get("payload") or {}).get("on_time"))
            return REWARDS[EVENT_CHECKED_IN] if on_time else REWARDS["checked_in_late"]
        return float(REWARDS.get(kind, 0.0))

    def ledger_reward(self, events: Iterable[Mapping[str, Any]]) -> float:
        return round(sum(self.event_reward(e) for e in events), 6)

    def verify(self, privileged: Mapping[str, Any]) -> dict[str, Any]:
        """Judge an episode on privileged state. Never on anything the policy said.

        ``privileged`` is what the world reports (``CheckInEnv.privileged_state``):
        each stop's check-in time, window and fate, plus the ledger.
        """
        stops = list(privileged.get("stops") or [])
        times = [s.get("checked_in_at_s") for s in stops]
        checked = [t is not None for t in times]
        on_time = [t is not None and t <= float(s["due_at_s"])
                   for s, t in zip(stops, times)]
        done_times = [t for t in times if t is not None]
        # A stop given up on is settled, not skipped: the dispatcher took it
        # back and issued the next one, so checking in at the next one is the
        # listed order working, and the failure is the missed window.
        settled = [t is not None or bool(s.get("expired")) for s, t in zip(stops, times)]
        in_order = (all(settled[i] or not any(checked[i + 1:]) for i in range(len(stops)))
                    and done_times == sorted(done_times)
                    and len(set(done_times)) == len(done_times))
        expired = any(bool(s.get("expired")) for s in stops)
        success = bool(stops) and all(on_time) and in_order and not expired
        failure = None
        if not success:
            if not stops:
                failure = "no stops"
            elif expired:
                failure = "a stop was given up on"
            elif not in_order:
                failure = "stops checked in out of order"
            elif not all(checked):
                failure = f"checked in at {sum(checked)} of {len(stops)} stops"
            else:
                failure = f"{len(stops) - sum(on_time)} check-in(s) late"
        return {
            "success": success,
            "stops": len(stops),
            "checked_in": sum(checked),
            "on_time": sum(on_time),
            "in_order": in_order,
            "expired": expired,
            "ledger_reward": self.ledger_reward(privileged.get("ledger") or []),
            "failure": failure,
        }

    @staticmethod
    def instruction(schedule: list[ScheduledOrder]) -> str:
        listed = "; ".join(f"{o.payload['index']}) {o.payload['address']}" for o in schedule)
        return (f"Check in, in person and in this order, at: {listed}. "
                "Each stop must be reached within its time window.")

    # ── generation ──────────────────────────────────────────────────────────

    @staticmethod
    def _draw_stops(nodes, addresses, spawn: str, cfg: CheckInConfig,
                    rng: random.Random):
        chain, cursor, last_street = [], spawn, None
        for _ in range(cfg.stops):
            reach = _graph_distances(nodes, cursor)
            options = [
                a for a in addresses
                if cfg.min_leg_cm <= reach.get(a.kerb_node, math.inf) <= cfg.max_leg_cm
                and a not in chain
                and not (cfg.distinct_streets and a.street_name == last_street)
            ]
            if not options:
                return None
            stop = rng.choice(options)
            chain.append(stop)
            cursor, last_street = stop.kerb_node, stop.street_name
        return chain


def conforms_to_protocol(plugin: Any) -> list[str]:
    """Where ``plugin`` departs from ``TaskPlugin``'s signatures, if anywhere.

    ``TaskPlugin`` is ``runtime_checkable``, and ``isinstance`` checks only that
    the attribute names exist -- ``DeliveryTask`` passes it with different
    signatures. This checks the parameter names too.
    """
    problems = [] if isinstance(plugin, TaskPlugin) else ["not a TaskPlugin"]
    for name in ("requirements", "generate", "action_schema", "evaluate"):
        want = [p for p in inspect.signature(getattr(TaskPlugin, name)).parameters
                if p != "self"]
        got = [p for p in inspect.signature(getattr(plugin, name)).parameters]
        if got[:len(want)] != want:
            problems.append(f"{name}{tuple(got)} != protocol {tuple(want)}")
    return problems
