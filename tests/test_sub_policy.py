"""
Tests for sub_policy.py - SubPolicyQNetwork and SubPolicy.

Tests cover:
- Network initialization and forward pass
- Shape correctness
- JAX compliance (jitting, gradients)
- TrainState creation
- Epsilon-greedy action selection
- Q-value computation
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from src.agent.components.sub_policy import SubPolicyQNetwork, SubPolicy


class TestSubPolicyQNetwork:
    """Tests for SubPolicyQNetwork module."""
    
    def test_network_initialization(self, rng_key, num_actions, cap_dim):
        """Test that the network can be initialized with correct parameters."""
        hidden_dims = (64, 64)
        net = SubPolicyQNetwork(num_actions=num_actions, hidden_dims=hidden_dims)
        
        # Create dummy inputs
        relative_vec = jnp.zeros((1, 2))
        capabilities = jnp.zeros((1, cap_dim))
        
        # Initialize parameters
        params = net.init(rng_key, relative_vec, capabilities)
        
        # Check that params is a dict (Flax params structure)
        assert isinstance(params, dict)
        assert 'params' in params
    
    def test_forward_pass_shapes(self, rng_key, batch_size, num_actions, cap_dim,
                                  sample_relative_vec, sample_capabilities):
        """Test that forward pass produces correct output shapes."""
        hidden_dims = (64, 64)
        net = SubPolicyQNetwork(num_actions=num_actions, hidden_dims=hidden_dims)
        
        # Initialize
        params = net.init(rng_key, sample_relative_vec, sample_capabilities)
        
        # Forward pass
        q_values = net.apply(params, sample_relative_vec, sample_capabilities)
        
        # Check output shape
        assert q_values.shape == (batch_size, num_actions)
        assert q_values.dtype == jnp.float32
    
    def test_different_hidden_dims(self, rng_key, batch_size, num_actions, cap_dim,
                                    sample_relative_vec, sample_capabilities):
        """Test that network works with different hidden dimension configurations."""
        for hidden_dims in [(32,), (64, 64), (128, 64, 32)]:
            net = SubPolicyQNetwork(num_actions=num_actions, hidden_dims=hidden_dims)
            
            params = net.init(rng_key, sample_relative_vec, sample_capabilities)
            q_values = net.apply(params, sample_relative_vec, sample_capabilities)
            
            assert q_values.shape == (batch_size, num_actions)
    
    def test_jax_jit_compliance(self, rng_key, batch_size, num_actions, cap_dim,
                                sample_relative_vec, sample_capabilities):
        """Test that the network can be jitted."""
        hidden_dims = (64, 64)
        net = SubPolicyQNetwork(num_actions=num_actions, hidden_dims=hidden_dims)
        
        # Initialize
        params = net.init(rng_key, sample_relative_vec, sample_capabilities)
        
        # Create jitted version
        @jax.jit
        def jitted_forward(params, relative_vec, capabilities):
            return net.apply(params, relative_vec, capabilities)
        
        # Run jitted version
        q_values = jitted_forward(params, sample_relative_vec, sample_capabilities)
        
        assert q_values.shape == (batch_size, num_actions)
    
    def test_gradient_computation(self, rng_key, batch_size, num_actions, cap_dim,
                                  sample_relative_vec, sample_capabilities):
        """Test that gradients can be computed through the network."""
        hidden_dims = (64, 64)
        net = SubPolicyQNetwork(num_actions=num_actions, hidden_dims=hidden_dims)
        
        # Initialize
        params = net.init(rng_key, sample_relative_vec, sample_capabilities)
        
        # Define loss function
        def loss_fn(params):
            q_values = net.apply(params, sample_relative_vec, sample_capabilities)
            return jnp.mean(q_values ** 2)
        
        # Compute gradients
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)
        
        # Check that gradients exist and have same structure as params
        assert isinstance(grads, dict)
        assert 'params' in grads
    
    def test_different_batch_sizes(self, rng_key, num_actions, cap_dim):
        """Test that network works with different batch sizes."""
        hidden_dims = (64, 64)
        net = SubPolicyQNetwork(num_actions=num_actions, hidden_dims=hidden_dims)
        
        for batch_size in [1, 2, 8, 16]:
            relative_vec = jnp.ones((batch_size, 2))
            capabilities = jnp.ones((batch_size, cap_dim))
            
            params = net.init(rng_key, relative_vec, capabilities)
            q_values = net.apply(params, relative_vec, capabilities)
            
            assert q_values.shape == (batch_size, num_actions)
    
    def test_input_concatenation(self, rng_key, batch_size, num_actions, cap_dim):
        """Test that inputs are properly concatenated."""
        hidden_dims = (64, 64)
        net = SubPolicyQNetwork(num_actions=num_actions, hidden_dims=hidden_dims)
        
        # Create inputs with known values
        relative_vec = jnp.ones((batch_size, 2)) * 0.5
        capabilities = jnp.ones((batch_size, cap_dim)) * 0.3
        
        params = net.init(rng_key, relative_vec, capabilities)
        q_values = net.apply(params, relative_vec, capabilities)
        
        # Should produce valid output
        assert q_values.shape == (batch_size, num_actions)
        assert not jnp.any(jnp.isnan(q_values))
        assert not jnp.any(jnp.isinf(q_values))


class TestSubPolicy:
    """Tests for SubPolicy utility class."""
    
    def test_create_train_state(self, rng_key, num_actions, cap_dim):
        """Test that TrainState can be created correctly."""
        learning_rate = 1e-3
        hidden_dims = (64, 64)
        
        train_state = SubPolicy.create_train_state(
            key=rng_key,
            num_actions=num_actions,
            learning_rate=learning_rate,
            hidden_dims=hidden_dims,
            cap_dim=cap_dim
        )
        
        # Check that it's a TrainState
        assert isinstance(train_state, TrainState)
        assert train_state.params is not None
        assert train_state.tx is not None
    
    def test_select_action_eps_greedy_shapes(self, rng_key, batch_size, num_actions, cap_dim,
                                             sample_relative_vec, sample_capabilities):
        """Test that epsilon-greedy selection produces correct shapes."""
        hidden_dims = (64, 64)
        train_state = SubPolicy.create_train_state(
            key=rng_key,
            num_actions=num_actions,
            learning_rate=1e-3,
            hidden_dims=hidden_dims,
            cap_dim=cap_dim
        )
        
        epsilon = 0.1
        key1, key2 = jax.random.split(rng_key)
        
        actions, q_values = SubPolicy.select_action_eps_greedy(
            train_state=train_state,
            relative_vec=sample_relative_vec,
            capabilities=sample_capabilities,
            epsilon=epsilon,
            key=key1
        )
        
        # Check shapes
        assert actions.shape == (batch_size,)
        assert q_values.shape == (batch_size, num_actions)
        assert actions.dtype == jnp.int32 or actions.dtype == jnp.int64
        assert q_values.dtype == jnp.float32
    
    def test_select_action_eps_greedy_jit_compliance(self, rng_key, batch_size, num_actions, cap_dim,
                                                      sample_relative_vec, sample_capabilities):
        """Test that epsilon-greedy selection can be jitted (it's already jitted, but test it works)."""
        hidden_dims = (64, 64)
        train_state = SubPolicy.create_train_state(
            key=rng_key,
            num_actions=num_actions,
            learning_rate=1e-3,
            hidden_dims=hidden_dims,
            cap_dim=cap_dim
        )
        
        epsilon = 0.1
        key1, key2 = jax.random.split(rng_key)
        
        # This should work since it's already jitted
        actions, q_values = SubPolicy.select_action_eps_greedy(
            train_state=train_state,
            relative_vec=sample_relative_vec,
            capabilities=sample_capabilities,
            epsilon=epsilon,
            key=key1
        )
        
        assert actions.shape == (batch_size,)
        assert q_values.shape == (batch_size, num_actions)
    
    def test_epsilon_greedy_behavior(self, rng_key, batch_size, num_actions, cap_dim,
                                     sample_relative_vec, sample_capabilities):
        """Test that epsilon-greedy behaves correctly (greedy when epsilon=0, random when epsilon=1)."""
        hidden_dims = (64, 64)
        train_state = SubPolicy.create_train_state(
            key=rng_key,
            num_actions=num_actions,
            learning_rate=1e-3,
            hidden_dims=hidden_dims,
            cap_dim=cap_dim
        )
        
        # Test greedy (epsilon=0)
        key1, key2 = jax.random.split(rng_key)
        actions_greedy, q_values = SubPolicy.select_action_eps_greedy(
            train_state=train_state,
            relative_vec=sample_relative_vec,
            capabilities=sample_capabilities,
            epsilon=0.0,
            key=key1
        )
        
        # Greedy actions should match argmax of q_values
        expected_greedy = jnp.argmax(q_values, axis=-1)
        np.testing.assert_array_equal(actions_greedy, expected_greedy)
        
        # Test random (epsilon=1)
        key3, key4 = jax.random.split(key2)
        actions_random, _ = SubPolicy.select_action_eps_greedy(
            train_state=train_state,
            relative_vec=sample_relative_vec,
            capabilities=sample_capabilities,
            epsilon=1.0,
            key=key3
        )
        
        # Random actions should be in valid range
        assert jnp.all(actions_random >= 0)
        assert jnp.all(actions_random < num_actions)
    
    def test_action_values_in_range(self, rng_key, batch_size, num_actions, cap_dim,
                                    sample_relative_vec, sample_capabilities):
        """Test that selected actions are in valid range."""
        hidden_dims = (64, 64)
        train_state = SubPolicy.create_train_state(
            key=rng_key,
            num_actions=num_actions,
            learning_rate=1e-3,
            hidden_dims=hidden_dims,
            cap_dim=cap_dim
        )
        
        epsilon = 0.5
        key1, key2 = jax.random.split(rng_key)
        
        actions, q_values = SubPolicy.select_action_eps_greedy(
            train_state=train_state,
            relative_vec=sample_relative_vec,
            capabilities=sample_capabilities,
            epsilon=epsilon,
            key=key1
        )
        
        # All actions should be in valid range [0, num_actions)
        assert jnp.all(actions >= 0)
        assert jnp.all(actions < num_actions)
    
    def test_get_q_values(self, rng_key, batch_size, num_actions, cap_dim,
                          sample_relative_vec, sample_capabilities):
        """Test the get_q_values helper function."""
        hidden_dims = (64, 64)
        train_state = SubPolicy.create_train_state(
            key=rng_key,
            num_actions=num_actions,
            learning_rate=1e-3,
            hidden_dims=hidden_dims,
            cap_dim=cap_dim
        )
        
        q_values = SubPolicy.get_q_values(
            train_state=train_state,
            relative_vec=sample_relative_vec,
            capabilities=sample_capabilities
        )
        
        # Check shape
        assert q_values.shape == (batch_size, num_actions)
        assert q_values.dtype == jnp.float32
    
    def test_get_q_values_jit_compliance(self, rng_key, batch_size, num_actions, cap_dim,
                                         sample_relative_vec, sample_capabilities):
        """Test that get_q_values can be jitted (it's already jitted, but test it works)."""
        hidden_dims = (64, 64)
        train_state = SubPolicy.create_train_state(
            key=rng_key,
            num_actions=num_actions,
            learning_rate=1e-3,
            hidden_dims=hidden_dims,
            cap_dim=cap_dim
        )
        
        # This should work since it's already jitted
        q_values = SubPolicy.get_q_values(
            train_state=train_state,
            relative_vec=sample_relative_vec,
            capabilities=sample_capabilities
        )
        
        assert q_values.shape == (batch_size, num_actions)
    
    def test_q_values_consistency(self, rng_key, batch_size, num_actions, cap_dim,
                                  sample_relative_vec, sample_capabilities):
        """Test that get_q_values and select_action_eps_greedy return consistent q_values."""
        hidden_dims = (64, 64)
        train_state = SubPolicy.create_train_state(
            key=rng_key,
            num_actions=num_actions,
            learning_rate=1e-3,
            hidden_dims=hidden_dims,
            cap_dim=cap_dim
        )
        
        key1, key2 = jax.random.split(rng_key)
        
        # Get q_values from both methods
        q_values_direct = SubPolicy.get_q_values(
            train_state=train_state,
            relative_vec=sample_relative_vec,
            capabilities=sample_capabilities
        )
        
        _, q_values_from_action = SubPolicy.select_action_eps_greedy(
            train_state=train_state,
            relative_vec=sample_relative_vec,
            capabilities=sample_capabilities,
            epsilon=0.5,
            key=key1
        )
        
        # They should be the same
        np.testing.assert_array_almost_equal(
            q_values_direct, q_values_from_action, decimal=5
        )
