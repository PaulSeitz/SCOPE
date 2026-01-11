"""
Pytest configuration and shared fixtures for testing JAX components.
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState


@pytest.fixture
def rng_key():
    """Provide a random key for JAX operations."""
    return jax.random.key(42)


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 4


@pytest.fixture
def num_concepts():
    """Number of concepts for meta policy."""
    return 5


@pytest.fixture
def num_actions():
    """Number of actions for sub policy."""
    return 18


@pytest.fixture
def max_objs():
    """Maximum number of objects."""
    return 35


@pytest.fixture
def global_dim():
    """Dimension of global features."""
    return 10


@pytest.fixture
def obj_dim():
    """Dimension of object features (x, y, w, h)."""
    return 4


@pytest.fixture
def cap_dim():
    """Dimension of capabilities."""
    return 4


@pytest.fixture
def sample_global_features(batch_size, global_dim):
    """Sample global features."""
    return jnp.ones((batch_size, global_dim), dtype=jnp.float32)


@pytest.fixture
def sample_objects(batch_size, max_objs, obj_dim):
    """Sample object features."""
    return jnp.ones((batch_size, max_objs, obj_dim), dtype=jnp.float32)


@pytest.fixture
def sample_object_labels(batch_size, max_objs):
    """Sample object labels."""
    return jnp.zeros((batch_size, max_objs), dtype=jnp.int32)


@pytest.fixture
def sample_object_mask(batch_size, max_objs):
    """Sample object mask (1 for real objects, 0 for padding)."""
    # First 10 objects are real, rest are padding
    mask = jnp.zeros((batch_size, max_objs), dtype=jnp.float32)
    mask = mask.at[:, :10].set(1.0)
    return mask


@pytest.fixture
def sample_relative_vec(batch_size):
    """Sample relative vector (dx, dy)."""
    return jnp.ones((batch_size, 2), dtype=jnp.float32)


@pytest.fixture
def sample_capabilities(batch_size, cap_dim):
    """Sample capabilities."""
    return jnp.ones((batch_size, cap_dim), dtype=jnp.float32)
