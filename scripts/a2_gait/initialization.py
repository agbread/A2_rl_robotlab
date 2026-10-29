"""Testable initialization boundary; loading is never an implicit scratch fallback."""
from common import BASE, PROTOCOL, fingerprint, recipe


def config_hash(state, num_envs):
    return fingerprint(dict(spec=recipe(state["branch"], state["phase"]), num_envs=num_envs,
                            seed=state["seed"], protocol=PROTOCOL))


def initialize(runner, state, num_envs):
    checkpoint = state.get("checkpoint")
    if state["branch"] == "scratch" and state["updates"] == 0:
        if checkpoint:
            raise ValueError("Initial scratch training must not load any checkpoint")
        return "random"
    if not checkpoint:
        if state["branch"] != "warm" or state["updates"]:
            raise ValueError("Continuation requires its own checkpoint")
        runner.load_weights(str(BASE))
        return "warm_start"
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = saved.get("gait_experiment", {})
    for key in ("branch", "seed", "updates", "transitions"):
        if metadata.get(key) != state[key]:
            raise ValueError(f"Checkpoint provenance/counters differ: {key}")
    if state.get("phase_entry", True):
        runner.load_weights(checkpoint)
        runner.alg.finetune_step = state["updates"]
        return "phase_transition"
    if metadata.get("config_hash") != config_hash(state, num_envs):
        raise ValueError("Resume configuration differs from the saved experiment")
    runner.load(checkpoint)
    return "resume"
