"""Pytest configuration and shared fixtures."""

import pytest
import jax


@pytest.fixture
def rng_key():
    """Provide a random key for JAX operations."""
    return jax.random.key(42)
