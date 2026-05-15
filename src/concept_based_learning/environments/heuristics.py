import jax.numpy as jnp


def get_nearest_object_target(raw_obs, objects, last_target, rng, *, no_target_idx=-1):
    """Select the closest active object to the player centre. Env-agnostic."""
    p_cx = raw_obs.player.x[-1] + raw_obs.player.width[-1] / 2.0
    p_cy = raw_obs.player.y[-1] + raw_obs.player.height[-1] / 2.0

    obj_cx = objects.x[-1] + objects.width[-1] / 2.0
    obj_cy = objects.y[-1] + objects.height[-1] / 2.0

    dist = jnp.sqrt((obj_cx - p_cx) ** 2 + (obj_cy - p_cy) ** 2)
    active = objects.active[-1] > 0.5
    dist_masked = jnp.where(active, dist, jnp.inf)

    nearest_idx = jnp.argmin(dist_masked)
    return jnp.where(jnp.any(active), nearest_idx, jnp.int32(no_target_idx))
