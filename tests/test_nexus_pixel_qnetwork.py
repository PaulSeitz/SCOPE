"""Hierarchical NEXUS pixel encoder matches the corrected PQN channel stack."""

from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pytest

# Optional external NEXUS dependency (third_party/symbolic_options plus
# third_party/patches/symbolic_options_thesis.patch).
pytest.importorskip("symbolic_options.hierarchical_pqn_jaxtari")

from concept_based_learning.observations.native_oc_stack import stack_pqn_pixel_channels
from symbolic_options.hierarchical_pqn_jaxtari import (
    CustomTrainState,
    QNetwork,
    optional_batch_stats,
    stack_pixel_channels,
)
from symbolic_options.purejaxql.save_load import load_params, save_params


def test_channel_stack_matches_pqn_and_keeps_spatial_order():
    images = jnp.arange(2 * 4 * 5 * 5 * 1, dtype=jnp.float32).reshape(2, 4, 5, 5, 1)
    assert jnp.array_equal(stack_pixel_channels(images), stack_pqn_pixel_channels(images))
    assert jnp.allclose(stack_pixel_channels(images)[0, 1, 2], images[0, :, 1, 2, 0])


def test_object_centric_network_stays_an_mlp():
    net = QNetwork(action_dim=6, hidden_size=64, num_layers=3, object_centric=True)
    variables = net.init(jax.random.PRNGKey(0), jnp.zeros((32,)), train=False)
    keys = set(variables["params"])
    assert "Dense_0" in keys
    assert not any(name.startswith("_PixelCNN") or name == "CNN" for name in keys)
    out = net.apply(variables, jnp.zeros((4, 32)), train=False)
    assert out.shape == (4, 6)


def test_pixel_network_uses_four_input_channels():
    net = QNetwork(action_dim=18, object_centric=False)
    images = jnp.zeros((1, 4, 84, 84, 1))
    variables = net.init(jax.random.PRNGKey(0), images, train=False)
    kernel = variables["params"]["_PixelCNN_0"]["Conv_0"]["kernel"]
    assert kernel.shape == (8, 8, 4, 32)
    dense = variables["params"]["_PixelCNN_0"]["Dense_0"]["kernel"]
    assert dense.shape[0] == 3136
    batch = jnp.zeros((3, 4, 84, 84, 1))
    out = net.apply(variables, batch, train=False)
    assert out.shape == (3, 18)
    assert optional_batch_stats(variables) == {}


def test_pixel_agent_inits_updates_and_reloads_without_batch_stats(tmp_path: Path):
    """The startup crash was ``network_variables['batch_stats']`` on LayerNorm."""
    net = QNetwork(action_dim=18, object_centric=False, norm_type="layer_norm")
    images = jnp.zeros((4, 4, 84, 84, 1))

    def create_agent(rng, params, batch_stats):
        variables = net.init(rng, images[:1], train=False)
        tx = optax.chain(optax.clip_by_global_norm(10.0), optax.radam(1e-4))
        return CustomTrainState.create(
            apply_fn=net.apply,
            params=variables["params"] if params is None else params,
            batch_stats=optional_batch_stats(variables) if batch_stats is None else batch_stats,
            tx=tx,
        )

    states = jax.vmap(create_agent, in_axes=(0, None, None))(
        jax.random.split(jax.random.PRNGKey(0), 3), None, None
    )
    assert states.batch_stats == {}
    state = create_agent(jax.random.PRNGKey(1), None, None)

    def loss(params, batch_stats):
        q_vals, updates = net.apply(
            {"params": params, "batch_stats": batch_stats},
            images,
            train=True,
            mutable=["batch_stats"],
        )
        return jnp.mean(q_vals ** 2), updates

    (loss_value, updates), grads = jax.value_and_grad(loss, has_aux=True)(
        state.params, state.batch_stats
    )
    assert jnp.isfinite(loss_value)
    new_state = state.apply_gradients(grads=grads)
    new_state = new_state.replace(
        batch_stats=updates.get("batch_stats", new_state.batch_stats)
    )
    assert new_state.batch_stats == {}

    params_path = tmp_path / "pqn_hier_learned_pixel_Seaquest_seed44_vmap0.safetensors"
    stats_path = tmp_path / "pqn_hier_learned_pixel_Seaquest_seed44_vmap0_bs.safetensors"
    save_params(new_state.params, params_path)
    save_params(new_state.batch_stats, stats_path)
    reloaded = load_params(params_path)
    reloaded_stats = load_params(stats_path) if stats_path.is_file() else None
    q = net.apply(
        {
            "params": reloaded,
            "batch_stats": reloaded_stats if reloaded_stats is not None else {},
        },
        images[:1],
        train=False,
    )
    assert q.shape == (1, 18)
    assert jnp.isfinite(q).all()
