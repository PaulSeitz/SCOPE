#!/usr/bin/env python3
"""Check a fresh NEXUS smoke checkpoint against its exact random initialization.

Usage: python -B scripts/verify_nexus_updates.py --ckpt tmp/nexus_repair_smoke/train
Requires a fresh training run (LOAD_PARAMS=false).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    args = parser.parse_args()
    from nexus.compat import prepare_nexus_runtime
    from nexus.paths import resolve_nexus_root, resolve_jaxatari_root
    prepare_nexus_runtime(nexus_root=resolve_nexus_root(), jaxatari_root=resolve_jaxatari_root())
    import jax
    import jax.numpy as jnp
    import numpy as np
    from omegaconf import OmegaConf
    from symbolic_options.hierarchical_pqn_jaxtari import QNetwork, lambda_return_targets
    from symbolic_options.purejaxql.save_load import load_params
    from nexus.mod_eval import _find_ckpt

    # Hand-computed Eq. 2 checks with nonconstant Q(s_t), including a terminal.
    r = jnp.array([1., 2., 3.])[:, None]
    v = jnp.array([10., 20., 30.])[:, None]
    boot = jnp.array([40.])
    for lam, expected in [(0., [19., 29., 39.]), (.5, [24.8725, 33.05, 39.]), (1., [34.39, 37.1, 39.])]:
        actual = lambda_return_targets(r, jnp.zeros_like(r), v, boot, .9, lam)
        np.testing.assert_allclose(actual[:, 0], expected, rtol=1e-6)
    terminal = lambda_return_targets(r, jnp.array([0., 1., 0.])[:, None], v, boot, .9, .5)
    np.testing.assert_allclose(terminal[:, 0], [10.9, 2., 39.], rtol=1e-6)

    ckpt = _find_ckpt(args.ckpt, "pqn_hier_llm", "Seaquest")
    configs = list(ckpt.parent.glob("*_config.yaml"))
    cfg = OmegaConf.to_container(OmegaConf.load(configs[0]), resolve=True)
    cfg = {**cfg, **cfg.get("alg", {})}
    assert not cfg.get("LOAD_PARAMS", False), "Requires fresh training"
    trained = load_params(ckpt)
    leaves = jax.tree.leaves(trained)
    assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
    net = QNetwork(
        action_dim=int(trained[f"Dense_{cfg['NUM_LAYERS']}"]["bias"].shape[-1]),
        hidden_size=cfg["HIDDEN_SIZE"], num_layers=cfg["NUM_LAYERS"],
        norm_type=cfg["NORM_TYPE"], norm_input=cfg.get("NORM_INPUT", False),
    )
    rng = jax.random.split(jax.random.PRNGKey(cfg["SEED"]), cfg["NUM_SEEDS"])[0]
    keys = jax.random.split(rng, cfg["NUM_AGENTS"])
    obs_size = int(trained["Dense_0"]["kernel"].shape[-2])
    initial = jax.vmap(lambda k: net.init(k, jnp.zeros(obs_size), train=False)["params"])(keys)
    deltas = [np.asarray(a - b) for a, b in zip(jax.tree.leaves(trained), jax.tree.leaves(initial))]
    max_changes = [max(float(np.max(np.abs(x[i]))) for x in deltas) for i in range(cfg["NUM_AGENTS"])]
    assert all(x > 0 for x in max_changes), max_changes
    summary = {
        "checkpoint": str(ckpt), "lambda_return_checks": "passed",
        "rollout_updates": int(cfg["TOTAL_TIMESTEPS"] // cfg["NUM_ENVS"] // cfg["NUM_STEPS"]),
        "gradient_steps_per_skill": int(cfg["TOTAL_TIMESTEPS"] // cfg["NUM_ENVS"] // cfg["NUM_STEPS"] * cfg["NUM_EPOCHS"] * cfg["NUM_MINIBATCHES"]),
        "max_parameter_change_per_skill": max_changes, "all_parameters_finite": True,
    }
    target = args.ckpt / "verification.json"
    target.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
