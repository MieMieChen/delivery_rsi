#!/usr/bin/env python3
"""Stage 4 of scripts/smoke/run_checkin_smoke.sh: everything GRPO needs, checked
before Ray and vLLM start, so a missing piece fails here with its name instead
of ten minutes into an engine start-up.

Reads its sizing from the environment the launcher exported (MODEL_PATH,
N_GPUS, PROMPT_LEN, RESP_LEN, DATASET_TRAIN, DATASET_VAL). Exits 1 on any hard
failure; prints WARN lines for things that will run but may truncate.
"""
from __future__ import annotations

import asyncio
import importlib
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "vendor/vagen")]

FAILURES: list[str] = []


def ok(message: str) -> None:
    print(f"ok    {message}", flush=True)


def warn(message: str) -> None:
    print(f"WARN  {message}", flush=True)


def bad(message: str) -> None:
    print(f"FAIL  {message}", flush=True)
    FAILURES.append(message)


def env_path(name: str) -> Path:
    value = os.environ[name]
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def check_imports() -> None:
    for name in ("torch", "vllm", "ray", "transformers", "flash_attn", "verl", "vagen",
                 "omegaconf", "hydra"):
        try:
            module = importlib.import_module(name)
            ok(f"import {name} {getattr(module, '__version__', '')}".rstrip())
        except Exception as error:  # noqa: BLE001
            bad(f"import {name}: {type(error).__name__}: {error}")


def check_model() -> object | None:
    model = Path(os.environ["MODEL_PATH"])
    if not (model / "config.json").exists():
        bad(f"MODEL_PATH {model} has no config.json")
        return None
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        bad(f"MODEL_PATH {model} has no *.safetensors weights")
    else:
        ok(f"model {model} ({len(shards)} shards, "
           f"{sum(s.stat().st_size for s in shards) / 1e9:.1f} GB)")
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(model))
    except Exception as error:  # noqa: BLE001
        bad(f"tokenizer from {model}: {error}")
        return None


def check_env_and_yamls(tokenizer: object | None) -> None:
    from vagen.envs.gym_image_env import GymImageEnv
    from vagen.gym_agent_dataset import _generate_seeds_for_spec, load_envspecs

    # Exactly how GymAgentLoop.init_class resolves a registry entry.
    module_name, class_name = "embodiedbench.training.vagen_task_env.TaskGymEnv".rsplit(".", 1)
    env_cls = getattr(importlib.import_module(module_name), class_name)
    if issubclass(env_cls, GymImageEnv):
        ok("TaskGymEnv is a vagen GymImageEnv")
    else:
        bad("TaskGymEnv is not a vagen GymImageEnv subclass")

    prompt_len = int(os.environ.get("PROMPT_LEN", "4608"))
    resp_len = int(os.environ.get("RESP_LEN", "12288"))
    for var in ("DATASET_TRAIN", "DATASET_VAL"):
        path = env_path(var)
        spec = load_envspecs(str(path)).specs[0]
        seeds = _generate_seeds_for_spec(spec, 0, 0)
        if spec.name != "TaskEnv":
            bad(f"{path.name}: env name {spec.name!r} is not the registered 'TaskEnv'")
        ok(f"{path.name}: {spec.n_envs} episodes, seeds {min(seeds)}-{max(seeds)}, "
           f"max_turns {spec.max_turns}, {spec.response_length_per_turn} tokens/turn")

        env = env_cls(env_config=dict(spec.config))
        obs, info = asyncio.run(env.reset(seed=seeds[0]))
        system = asyncio.run(env.system_prompt())["obs_str"]
        if "<image>" in obs["obs_str"]:
            bad(f"{path.name}: the first observation carries an image placeholder")
        steps = ['THOUGHT: look around.\n```\nlook()\n```',
                 'THOUGHT: try.\n```\ncheck_in()\n```',
                 'THOUGHT: wait.\n```\nwait()\n```']
        for reply in steps:
            o, reward, done, step_info = asyncio.run(env.step(reply))
            if done:
                break
        asyncio.run(env.close())
        ok(f"{path.name}: reset/system_prompt/step ran on seed {seeds[0]} "
           f"({info['instance_id']}, last status {step_info['status']})")
        if tokenizer is None:
            continue
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": obs["obs_str"]}]
        first = len(tokenizer.apply_chat_template(messages, tokenize=True,
                                                  add_generation_prompt=True))
        # A whole episode, walking: the notes grow as streets are walked, so
        # the first few observations understate the last ones.
        walked = walk_episode(env_cls, spec.config, seeds[0], spec.max_turns)
        sizes = [len(tokenizer(t)["input_ids"]) + 16 for t in walked]
        budget = sum(sizes) + spec.max_turns * spec.response_length_per_turn
        line = (f"{path.name}: prompt {first} tokens (PROMPT_LEN {prompt_len}); "
                f"observations {min(sizes)}-{max(sizes)} tokens over a {len(walked)}-turn walk, "
                f"worst case {budget} with full-length replies (RESP_LEN {resp_len})")
        if first > prompt_len:
            bad(line + " -- the prompt does not fit")
        elif budget > resp_len:
            warn(line + " -- long episodes will be cut at RESP_LEN")
        else:
            ok(line)


STREET = re.compile(r'"([^"]+)" — going ([a-z-]+)')


def walk_episode(env_cls: type, config: dict, seed: int, max_turns: int) -> list[str]:
    """Every observation of one episode that walks a different street each turn."""
    env = env_cls(env_config=dict(config))
    obs, _ = asyncio.run(env.reset(seed=seed))
    text, texts = obs["obs_str"], []
    for turn in range(max_turns):
        streets = STREET.findall(text)
        if streets:
            street, heading = streets[turn % len(streets)]
            reply = f'THOUGHT: walk on.\n```\nwalk_to("{street}", "{heading}")\n```'
        else:
            reply = "THOUGHT: wait.\n```\nwait()\n```"
        o, _, done, _ = asyncio.run(env.step(reply))
        text = o["obs_str"]
        texts.append(text)
        if done:
            break
    asyncio.run(env.close())
    return texts or [obs["obs_str"]]


def check_gpus() -> None:
    try:
        import torch
    except ImportError as error:
        bad(f"GPU check needs torch: {error}")
        return

    want = int(os.environ.get("N_GPUS", "4"))
    seen = torch.cuda.device_count()
    usable = 0
    for index in range(seen):
        try:
            torch.zeros(8, device=f"cuda:{index}")
            free, total = torch.cuda.mem_get_info(index)
            print(f"      cuda:{index} {torch.cuda.get_device_name(index)} "
                  f"{free / 2**30:.1f}/{total / 2**30:.1f} GiB free", flush=True)
            usable += 1
        except Exception as error:  # noqa: BLE001
            print(f"      cuda:{index} unusable: {error}", flush=True)
    if usable == want:
        ok(f"{usable} usable GPU(s), N_GPUS={want}")
    else:
        bad(f"{usable} usable GPU(s) of {seen} visible, but N_GPUS={want} "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")


def check_disk() -> None:
    out = Path(os.environ.get("EXPERIMENT_DIR", REPO))
    free = shutil.disk_usage(out).free / 2**30
    (ok if free > 20 else warn)(f"{free:.0f} GiB free under {out}")


def main() -> int:
    check_imports()
    tokenizer = check_model()
    if not FAILURES:
        check_env_and_yamls(tokenizer)
    check_gpus()
    check_disk()
    if FAILURES:
        print(f"\npreflight FAILED ({len(FAILURES)}):", *FAILURES, sep="\n  ")
        return 1
    print("\npreflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
