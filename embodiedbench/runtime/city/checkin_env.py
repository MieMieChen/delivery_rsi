"""The world that executes a ``checkin`` episode (``embodiedbench/tasks/checkin.py``).

``CheckInEnv`` is the courier city with a different job: the same streets,
walking, clock, lights and stamina, but no parcels. The episode's stops come
from the task's ``EpisodeSpec`` and are never re-drawn here, so the spec's hash
describes the episode actually played.

Two additions the generated tasks of the proposal will also need:

* **An append-only event ledger.** Every arrival at the current stop, every
  check-in and every refused check-in is written once, as a
  ``schemas.runtime.Event``, and never edited. The task's reward is a function
  of these events alone, and the verifier reads them back.
* **A task-specific Tool.** ``check_in`` replaces ``collect``/``hand_over`` in
  the menu. It is implemented on the courier's own primitives (the door
  tolerance, the refusal charge, the queue), so it inherits their contracts.

The reference couriers (``tasks/courier_oracle.py``, ``tasks/courier_router.py``)
drive the world through ``hand_over``; here that name is kept as an alias of
``check_in`` so the floor and ceiling run unchanged. Each stop is an ``Order``
whose pickup and drop-off are the same address and which starts collected, and
the queue depth is pinned at one, which is what enforces the listed order.
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

from embodiedbench.agent.courier.session import CourierSession
from embodiedbench.runtime.city.courier_env import (
    HANDLING_SECONDS,
    TIME_BUDGET_MULTIPLE,
    CourierEnv,
    Order,
    StepOutcome,
)
from embodiedbench.schemas.runtime import Event
from embodiedbench.tasks.checkin import (
    CHECK_IN,
    EVENT_ARRIVED,
    EVENT_CHECKED_IN,
    EVENT_REFUSED,
)

#: Tools of the delivery job that mean nothing here. ``check_order`` reads the
#: slip's pickup, drop-off and fee; the stop list is in the clock block instead.
_DELIVERY_TOOLS = frozenset({"collect", "hand_over", "check_order", "accept_job"})


class CheckInEnv(CourierEnv):
    """A courier city whose job is a fixed list of check-ins."""

    def __init__(self, network: Any, *, stops: Sequence[str],
                 spawn_node: str | None = None, episode_id: str = "checkin",
                 deadline_slack: float = 2.5, **kwargs: Any):
        if not stops:
            raise ValueError("a check-in episode needs at least one stop")
        # A difficulty tier would choose its own order count, queue depth and
        # clock -- and ENDLESS would draw delivery orders when the list ran
        # out. The task chooses all three here, so the tier is dropped.
        kwargs.pop("difficulty", None)
        kwargs.pop("order_bias", None)
        kwargs.pop("order_count", None)
        kwargs["queue_depth"] = 1
        super().__init__(network, order_count=len(stops),
                         deadline_slack=deadline_slack, **kwargs)
        by_id = {a.building_id: a for a in network.addresses}
        missing = [s for s in stops if s not in by_id]
        if missing:
            raise ValueError(f"unknown stop sites {missing}")
        self.stop_addresses = [by_id[s] for s in stops]
        self._spawn_override = spawn_node
        self.episode_id = episode_id
        self.ledger: list[dict[str, Any]] = []
        self._arrived: set[int] = set()

    @classmethod
    def from_episode(cls, network: Any, spec: Any, **kwargs: Any) -> "CheckInEnv":
        """The world for one ``EpisodeSpec``: its stops, its spawn, its seed."""
        x, y = spec.spawn.position.x_cm, spec.spawn.position.y_cm
        spawn = min(network.nodes, key=lambda n: math.dist(network.nodes[n].position, (x, y)))
        kwargs.setdefault("seed", spec.seed)
        return cls(network,
                   stops=[o.pickup_site for o in spec.order_schedule],
                   spawn_node=spawn, episode_id=spec.instance_id, **kwargs)

    # ── the episode ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.ledger = []
        self._arrived = set()
        super().reset()
        # The tier used to set this. One rule, the courier's own: the work the
        # list actually needs, times the room a policy may waste.
        self.shift_seconds = self.optimal_seconds * TIME_BUDGET_MULTIPLE
        self.poll_events()

    def _choose_spawn_node(self, rng: Any) -> str:
        if self._spawn_override is not None:
            return self._spawn_override
        return super()._choose_spawn_node(rng)

    def _make_orders(self, rng: Any) -> list[Order]:
        orders, cursor = [], self.node_id
        for index, address in enumerate(self.stop_addresses, start=1):
            leg = self.route_cost(cursor, address.kerb_node, obstacles=True)
            if leg is None:
                raise ValueError(f"stop {index} ({address.text}) is unreachable")
            orders.append(Order(
                index=index, pickup=address, dropoff=address, fee=1.0,
                deadline_s=leg[0] * self.deadline_slack + HANDLING_SECONDS,
                picked_up=True))
            cursor = address.kerb_node
        return orders

    @property
    def shift_over(self) -> bool:
        # Also over once every stop is settled and one of them was given up on,
        # so such an episode ends instead of walking on with nothing to do. A
        # list that was simply finished ends through ``finished`` instead.
        orders = getattr(self, "orders", None)
        abandoned = (bool(orders) and all(o.delivered or o.expired for o in orders)
                     and any(o.expired for o in orders))
        return super().shift_over or abandoned

    # ── the task-specific action ─────────────────────────────────────────────

    def check_in(self) -> StepOutcome:
        # Standing at the door is an arrival whoever is driving: the training
        # env polls after every step, the reference couriers do not, and their
        # ledgers must price the same episode the same way.
        self.poll_events()
        order = self._order_at_hand(collected=True)
        if order is None:
            current = self.active_order()
            self._record(EVENT_REFUSED, {
                "stop": current.index if current else None,
                "gap_m": (round(math.dist(self.position(), current.dropoff.kerb) / 100.0, 1)
                          if current else None)})
            if current is None:
                return self._refuse(StepOutcome(
                    ok=False, code="no_stop", message="Your list is done; there is nowhere left to check in."))
            return self._refuse(StepOutcome(
                ok=False, code="not_at_stop",
                message=f"You are not at {current.dropoff.text}, the current stop on your list."))
        self.turns += 1
        self.sim_seconds += HANDLING_SECONDS + self.embodiment.stop_overhead_s
        order.delivered = True
        order.delivered_at_s = self.sim_seconds
        on_time = self.sim_seconds <= order.due_at()
        order.paid = order.fee if on_time else 0.0
        self.earnings += order.paid
        self._record(EVENT_CHECKED_IN, {"stop": order.index, "address": order.dropoff.text,
                                        "on_time": on_time})
        self._issue()
        self._light_the_screen(reclaim=True)
        self.finished = not self.live_orders() or self.shift_over
        upcoming = self.active_order()
        tail = (f" Next on your list: {upcoming.dropoff.text}." if upcoming
                else " That was the last stop on your list.")
        self.poll_events()
        return StepOutcome(
            ok=True, sim_seconds=HANDLING_SECONDS, reward=1.0 if on_time else 0.0,
            finished=self.finished,
            message=f"You check in at {order.dropoff.text}{'' if on_time else ', late'}.{tail}")

    def hand_over(self) -> StepOutcome:
        """The reference couriers' name for the door action."""
        return self.check_in()

    # ── the ledger ───────────────────────────────────────────────────────────

    def _record(self, kind: str, payload: dict[str, Any]) -> None:
        event = Event(
            event_id=f"{self.episode_id}_ev_{len(self.ledger):05d}",
            episode_id=self.episode_id,
            step_index=int(self.turns),
            kind=kind,
            sim_time_s=float(self.sim_seconds),
            payload=payload,
            privileged=True,
        )
        self.ledger.append(event.model_dump(mode="json"))

    def poll_events(self) -> None:
        """Write the events the world can only notice by looking: arrivals.

        Called after every action by whoever drives the world. Arrival at the
        *current* stop only, once per stop, so walking back and forth past a
        door is not a source of events.
        """
        current = self.active_order()
        if current is None or current.index in self._arrived:
            return
        if math.dist(self.position(), current.dropoff.kerb) <= self._task_action_tolerance_cm(collected=True):
            self._arrived.add(current.index)
            self._record(EVENT_ARRIVED, {"stop": current.index, "address": current.dropoff.text})

    def privileged_state(self) -> dict[str, Any]:
        """What the verifier reads. Never shown to the policy."""
        return {
            "episode_id": self.episode_id,
            "stops": [{
                "index": o.index,
                "address": o.dropoff.text,
                "site": o.dropoff.building_id,
                "checked_in_at_s": o.delivered_at_s,
                "issued_at_s": o.issued_at_s,
                "due_at_s": (o.due_at() if o.issued_at_s is not None else None),
                "expired": o.expired,
            } for o in self.orders],
            "ledger": list(self.ledger),
            "sim_seconds": self.sim_seconds,
            "shift_seconds": self.shift_seconds,
            "turns": self.turns,
        }

    # ── what the courier is offered and shown ────────────────────────────────

    def allowed_tool_names(self) -> list[str]:
        names = [n for n in super().allowed_tool_names() if n not in _DELIVERY_TOOLS]
        return [CHECK_IN.name, *names]

    def tools_for_prompt(self) -> list[Any]:
        from embodiedbench.agent.courier.tools import TOOLS_BY_NAME

        limits = self.tool_limits()
        return [CHECK_IN if name == CHECK_IN.name else TOOLS_BY_NAME[name].filled(**limits)
                for name in self.allowed_tool_names()]

    def clock_text(self) -> str:
        """The whole list, each stop with its state, the current one timed."""
        if not self.orders:
            return ""
        lines = ["Your check-in list, in order:"]
        for order in self.orders:
            if order.delivered:
                state = "done" + ("" if order.paid else ", late")
            elif order.expired:
                state = "missed"
            elif order.live:
                left = order.minutes_left(self.sim_seconds)
                state = ("CURRENT, " + (f"{left:.0f} min left" if left >= 0
                                        else f"{abs(left):.0f} min overdue"))
            else:
                state = "later"
            lines.append(f"  {order.index}) {order.dropoff.text} — {state}")
        return "\n".join(lines)


#: The delivery job's opening, as ``prompts.SYSTEM_TEMPLATE`` renders it.
_DELIVERY_OPENING = re.compile(
    r"You are a delivery courier working on foot in (?P<city>.+?)\. You collect parcels and\s+"
    r"hand them to customers at street addresses, against a clock\.")


class CheckInSession(CourierSession):
    """The courier harness, told the job is a check-in list, not deliveries."""

    def system_prompt(self) -> str:
        text = super().system_prompt()
        opening = (
            "You are a courier working on foot in {city}. Today there are no parcels: "
            "you must check in, in person, at a list of street addresses, in the listed "
            "order, each within its time window. Call check_in() at a stop only when the "
            "street and door number match the current stop on your list.")
        replaced, count = _DELIVERY_OPENING.subn(
            lambda m: opening.format(city=m.group("city")), text, count=1)
        if count != 1:
            raise RuntimeError("the courier system prompt changed; update CheckInSession")
        return replaced

    def _set_goal(self) -> None:
        order = self.env.active_order()
        if order is None:
            self.memory.set_goal("", "")
        else:
            self.memory.set_goal("check in at", order.target.text)
