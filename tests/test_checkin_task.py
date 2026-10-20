"""The ``checkin`` smoke task: plugin, world, ledger, verifier, references, training env.

CPU only, tracked data only (the vendored Paris map), no albums. The parts that
need VAGEN (the GymImageEnv base class, VAGEN's yaml loader) skip without it.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city.checkin_env import CheckInEnv, CheckInSession
from embodiedbench.runtime.city.courier_env import CourierEnv
from embodiedbench.tasks.checkin import (
    CHECK_IN,
    EVENT_ARRIVED,
    EVENT_CHECKED_IN,
    EVENT_REFUSED,
    REWARDS,
    CheckInTask,
    conforms_to_protocol,
)

REPO = Path(__file__).resolve().parents[1]
MAPS = REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"
TRAIN_YAML = REPO / "experiments/smoke_checkin_train.yaml"
VAL_YAML = REPO / "experiments/smoke_checkin_val.yaml"
SMOKE_CONFIG = {"stops": 2, "max_leg_cm": 12000, "step_budget": 20}


@pytest.fixture(scope="module")
def paris():
    return build_road_network(MAPS, map_name="citycore-paris")


@pytest.fixture(scope="module")
def task():
    return CheckInTask(SMOKE_CONFIG)


def world_for(paris, task, seed, **kwargs):
    spec = task.generate(paris, seed)
    world = CheckInEnv.from_episode(paris, spec, stride="block", **task.world_kwargs(), **kwargs)
    world.reset()
    return spec, world


def stand_at(world, order):
    """Put the courier on a stop's kerb node -- a test-only teleport."""
    world.node_id = order.dropoff.kerb_node
    world.arrived_from = None


def run(coro):
    return asyncio.run(coro)


class TestPlugin:
    def test_it_matches_the_protocol_signatures_not_just_the_names(self, task):
        assert conforms_to_protocol(task) == []

    def test_generation_is_deterministic_in_process(self, paris, task):
        assert task.generate(paris, 7).content_hash() == task.generate(paris, 7).content_hash()
        assert task.generate(paris, 7).content_hash() != task.generate(paris, 8).content_hash()

    def test_generation_is_deterministic_across_processes(self, paris, task):
        code = (
            "from embodiedbench.compiler.road_network import build_road_network\n"
            "from embodiedbench.tasks.checkin import CheckInTask\n"
            f"net = build_road_network({str(MAPS)!r}, map_name='citycore-paris')\n"
            f"print(CheckInTask({SMOKE_CONFIG!r}).generate(net, 7).content_hash())\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
        env["PYTHONHASHSEED"] = "12345"
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                             capture_output=True, text=True, check=True).stdout
        assert out.strip().splitlines()[-1] == task.generate(paris, 7).content_hash()

    def test_the_spec_pins_stops_spawn_and_config(self, paris, task):
        spec = task.generate(paris, 3)
        assert spec.task.plugin == "checkin"
        assert spec.task.config_sha256 == task.config.sha256()
        sites = [o.pickup_site for o in spec.order_schedule]
        assert len(sites) == 2 and len(set(sites)) == 2
        assert all(o.pickup_site == o.dropoff_site for o in spec.order_schedule)
        streets = [o.payload["address"].split(" ", 1)[1] for o in spec.order_schedule]
        assert streets[0] != streets[1]
        assert "check in" in spec.observable_instruction.lower()

    def test_every_generated_spec_is_statically_solvable(self, paris, task):
        for seed in range(16):
            assert task.check_solvable(task.generate(paris, seed), paris).feasible, seed

    def test_a_budget_too_small_for_the_list_is_refused(self, paris, task):
        spec = task.generate(paris, 0)
        tight = spec.model_copy(update={"budgets": spec.budgets.model_copy(update={"steps": 5})})
        verdict = task.check_solvable(tight, paris)
        assert not verdict.feasible and "step budget" in verdict.reasons[0]

    def test_an_unknown_config_key_is_an_error_not_a_default(self):
        with pytest.raises(ValueError, match="unknown checkin config keys"):
            CheckInTask({"stopz": 3})


class TestWorld:
    def test_the_world_plays_the_spec_it_was_given(self, paris, task):
        spec, world = world_for(paris, task, 4)
        assert [o.dropoff.building_id for o in world.orders] == [
            o.pickup_site for o in spec.order_schedule]
        assert world.position(world.spawn_node) == pytest.approx(
            (spec.spawn.position.x_cm, spec.spawn.position.y_cm))
        assert [o.live for o in world.orders] == [True, False]

    def test_the_menu_is_the_tasks_not_the_delivery_jobs(self, paris, task):
        _, world = world_for(paris, task, 4)
        names = world.allowed_tool_names()
        assert names[0] == CHECK_IN.name
        assert not {"collect", "hand_over", "check_order"} & set(names)
        prompt = CheckInSession(world, city="Paris").system_prompt()
        assert "check_in(" in prompt
        assert "hand_over(" not in prompt and "collect(" not in prompt
        assert "delivery courier" not in prompt

    def test_a_refused_check_in_costs_a_turn_and_is_written_down(self, paris, task):
        _, world = world_for(paris, task, 4)
        outcome = world.check_in()
        assert not outcome.ok and outcome.code == "not_at_stop"
        assert world.turns == 1 and world.sim_seconds == pytest.approx(5.0)
        assert world.ledger[-1]["kind"] == EVENT_REFUSED

    def test_checking_in_at_the_current_stop_pays_and_advances(self, paris, task):
        _, world = world_for(paris, task, 4)
        first, second = world.orders
        stand_at(world, first)
        outcome = world.check_in()
        assert outcome.ok and first.delivered and second.live
        kinds = [e["kind"] for e in world.ledger]
        assert kinds == [EVENT_ARRIVED, EVENT_CHECKED_IN]
        assert world.ledger[-1]["payload"]["on_time"] is True

    def test_the_listed_order_is_enforced(self, paris, task):
        _, world = world_for(paris, task, 4)
        stand_at(world, world.orders[1])
        assert world.check_in().code == "not_at_stop"
        assert not world.orders[1].delivered

    def test_finishing_the_list_finishes_the_episode(self, paris, task):
        _, world = world_for(paris, task, 4)
        for order in world.orders:
            stand_at(world, order)
            assert world.check_in().ok
        assert world.finished and not world.shift_over
        verdict = task.verify(world.privileged_state())
        assert verdict["success"] and verdict["ledger_reward"] == pytest.approx(2.4)

    def test_the_ledger_is_append_only_with_unique_ids(self, paris, task):
        _, world = world_for(paris, task, 4)
        world.check_in()
        stand_at(world, world.orders[0])
        world.check_in()
        ids = [e["event_id"] for e in world.ledger]
        assert len(ids) == len(set(ids))
        steps = [e["step_index"] for e in world.ledger]
        assert steps == sorted(steps)
        assert all(e["privileged"] for e in world.ledger)

    def test_a_difficulty_tier_cannot_turn_it_back_into_deliveries(self, paris, task):
        spec = task.generate(paris, 4)
        world = CheckInEnv.from_episode(paris, spec, difficulty="endless", stride="block")
        world.reset()
        assert len(world.orders) == 2 and world.queue_depth == 1 and not world.unbounded


class TestVerifier:
    @staticmethod
    def state(*stops, ledger=()):
        return {"stops": [{"index": i, "checked_in_at_s": t, "due_at_s": due,
                           "expired": exp} for i, (t, due, exp) in enumerate(stops, 1)],
                "ledger": list(ledger)}

    def test_all_stops_on_time_and_in_order_is_success(self, task):
        assert task.verify(self.state((50, 100, False), (90, 200, False)))["success"]

    @pytest.mark.parametrize("stops, failure", [
        (((50, 100, False), (None, 200, False)), "checked in at 1 of 2"),
        (((150, 100, False), (190, 300, False)), "late"),
        (((None, 100, False), (90, 200, False)), "out of order"),
        (((90, 100, False), (50, 200, False)), "out of order"),
        (((50, 100, False), (None, 200, True)), "given up on"),
        (((None, 100, True), (90, 200, False)), "given up on"),
    ])
    def test_every_way_to_fail_is_caught_and_named(self, task, stops, failure):
        verdict = task.verify(self.state(*stops))
        assert not verdict["success"] and failure in verdict["failure"]

    def test_a_stop_given_up_on_is_not_an_ordering_violation(self, task):
        verdict = task.verify(self.state((None, 100, True), (90, 200, False)))
        assert verdict["in_order"] and verdict["expired"]

    def test_the_world_reports_a_missed_window_as_missed(self, paris, task):
        _, world = world_for(paris, task, 1000)
        first, second = world.orders
        while not first.expired:
            world.wait()
        assert second.live
        stand_at(world, second)
        assert world.check_in().ok
        verdict = task.verify(world.privileged_state())
        assert verdict["failure"] == "a stop was given up on" and verdict["in_order"]

    def test_reward_is_a_function_of_ledger_events_only(self, task):
        events = [{"kind": EVENT_ARRIVED}, {"kind": EVENT_CHECKED_IN, "payload": {"on_time": True}},
                  {"kind": EVENT_CHECKED_IN, "payload": {"on_time": False}},
                  {"kind": EVENT_REFUSED}, {"kind": "something_else"}]
        assert task.ledger_reward(events) == pytest.approx(
            REWARDS[EVENT_ARRIVED] + REWARDS[EVENT_CHECKED_IN] + REWARDS["checked_in_late"]
            + REWARDS[EVENT_REFUSED])


class TestReferences:
    def test_the_ceiling_solves_every_validation_seed_and_the_floor_does_not_beat_it(self, tmp_path):
        from tools import checkin_reference

        assert checkin_reference.main(["--yaml", str(VAL_YAML), "--out-dir", str(tmp_path)]) == 0
        floor = json.loads((tmp_path / "floor.json").read_text())
        ceiling = json.loads((tmp_path / "ceiling.json").read_text())
        assert ceiling["success_rate"] == 1.0
        assert floor["ledger_reward_mean"] <= ceiling["ledger_reward_mean"]
        assert floor["turns_mean"] >= ceiling["turns_mean"]
        assert floor["episodes"] == ceiling["episodes"] == 8


class TestTrainingEnv:
    @staticmethod
    def block(path):
        return yaml.safe_load(path.read_text())["envs"][0]

    def env(self, path=VAL_YAML):
        from embodiedbench.training.vagen_task_env import TaskGymEnv

        return TaskGymEnv(dict(self.block(path)["config"]))

    def test_the_yamls_name_the_registered_env_and_agree_on_the_task(self):
        train, val = self.block(TRAIN_YAML), self.block(VAL_YAML)
        assert train["name"] == val["name"] == "TaskEnv"
        assert train["config"]["task_config"] == val["config"]["task_config"]
        assert train["max_turns"] == train["config"]["max_turns"] == val["max_turns"]
        assert set(range(0, 128)).isdisjoint(range(1000, 1008))
        assert "album_root" not in train["config"] and "album_root" not in val["config"]

    def test_reward_comes_from_the_ledger_and_success_from_the_verifier(self):
        env = self.env()
        obs, info = run(env.reset(seed=1000))
        assert "<image>" not in obs["obs_str"] and "multi_modal_input" not in obs
        assert info["task"] == "checkin"
        prompt = run(env.system_prompt())["obs_str"]
        assert "check_in(" in prompt

        _, reward, done, info = run(env.step("THOUGHT: try.\n```\ncheck_in()\n```"))
        assert reward == pytest.approx(REWARDS[EVENT_REFUSED]) and not done
        assert info["status"] == "rejected" and not info["success"]

        for n, order in enumerate(env.world.orders, 1):
            stand_at(env.world, order)
            _, reward, done, info = run(env.step("THOUGHT: here.\n```\ncheck_in()\n```"))
            assert reward == pytest.approx(REWARDS[EVENT_ARRIVED] + REWARDS[EVENT_CHECKED_IN])
            assert info["checked_in"] == n
        assert done and info["success"] and info["termination"] == "delivered"

    def test_the_episode_log_gets_one_line_per_finished_episode(self, tmp_path):
        from embodiedbench.training.vagen_task_env import TaskGymEnv

        log = tmp_path / "episodes.jsonl"
        config = {**self.block(VAL_YAML)["config"], "task_episode_log": str(log), "max_turns": 2}
        env = TaskGymEnv(config)
        run(env.reset(seed=1001))
        run(env.step("THOUGHT: a.\n```\nwait()\n```"))
        _, _, done, _ = run(env.step("THOUGHT: b.\n```\nwait()\n```"))
        assert done
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        assert len(rows) == 1 and rows[0]["seed"] == 1001 and rows[0]["success"] is False

    def test_an_episode_cut_by_the_agent_loop_is_still_logged_once(self, tmp_path):
        from embodiedbench.training.vagen_task_env import TaskGymEnv

        log = tmp_path / "episodes.jsonl"
        env = TaskGymEnv({**self.block(VAL_YAML)["config"], "task_episode_log": str(log)})
        run(env.reset(seed=1002))
        run(env.step("THOUGHT: a.\n```\nwait()\n```"))
        run(env.close())
        run(env.close())
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        assert len(rows) == 1 and rows[0]["termination"] == "agent_loop_cut"

    def test_an_unknown_task_is_refused(self):
        from embodiedbench.training.vagen_task_env import TaskGymEnv

        with pytest.raises(ValueError, match="unknown task"):
            TaskGymEnv({"task": "nope"})

    def test_the_courier_env_still_builds_the_courier_world(self):
        from embodiedbench.training.vagen_courier_env import CourierGymEnv

        env = CourierGymEnv({"difficulty": "solo", "stride": "block"})
        run(env.reset(seed=3))
        assert type(env.world) is CourierEnv

    def test_it_is_a_real_vagen_env_when_vagen_is_present(self):
        pytest.importorskip("vagen.envs.gym_image_env")
        from vagen.envs.gym_image_env import GymImageEnv

        from embodiedbench.training.vagen_task_env import TaskGymEnv

        assert issubclass(TaskGymEnv, GymImageEnv)

    def test_vagens_own_loader_reads_both_yamls(self):
        pytest.importorskip("torch")
        pytest.importorskip("omegaconf")
        sys.path.insert(0, str(REPO / "vendor/vagen"))
        try:
            from vagen.gym_agent_dataset import _generate_seeds_for_spec, load_envspecs
        finally:
            sys.path.remove(str(REPO / "vendor/vagen"))
        for path in (TRAIN_YAML, VAL_YAML):
            spec = load_envspecs(str(path)).specs[0]
            seeds = _generate_seeds_for_spec(spec, 0, 0)
            assert len(seeds) == spec.n_envs == len(set(seeds))


def test_the_smoke_launcher_is_not_cut_short_by_a_comment():
    """Same rule as the trainer launcher: a comment inside a continued command
    ends it there (tests/test_courier_training.py)."""
    for path in sorted((REPO / "scripts/smoke").glob("*.sh")):
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines[:-1]):
            if line.rstrip().endswith("\\"):
                assert not lines[i + 1].strip().startswith("#"), f"{path.name}:{i + 2}"
