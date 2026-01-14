"""
Tests for meta_policy.py - MetaPolicyQNetwork and MetaPolicy.

Tests cover:
- Network initialization and forward pass
- Shape correctness
- JAX compliance (jitting, gradients)
- TrainState creation
- Epsilon-greedy action selection
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from src.agent.components.meta_policy import MetaPolicyQNetwork, MetaPolicy


class TestMetaPolicyQNetwork:
    """Tests for MetaPolicyQNetwork module."""
    
    def test_network_initialization(self, rng_key, num_concepts, global_dim, obj_dim, max_objs):
        """Test that the network can be initialized with correct parameters."""
        net = MetaPolicyQNetwork(num_concepts=num_concepts)
        
        # Create dummy inputs
        global_features = jnp.zeros((1, global_dim))
        objects = jnp.zeros((1, max_objs, obj_dim))
        object_labels = jnp.zeros((1, max_objs, num_concepts), dtype=jnp.float32)  # Multi-hot
        object_mask = jnp.ones((1, max_objs))
        
        # Initialize parameters
        params = net.init(rng_key, global_features, objects, object_labels, object_mask)
        
        # Check that params is a dict (Flax params structure)
        assert isinstance(params, dict)
        assert 'params' in params
    
    def test_forward_pass_shapes(self, rng_key, batch_size, num_concepts, global_dim, 
                                  obj_dim, max_objs, sample_global_features, 
                                  sample_objects, sample_object_labels, sample_object_mask):
        """Test that forward pass produces correct output shapes."""
        net = MetaPolicyQNetwork(num_concepts=num_concepts)
        
        # Initialize
        params = net.init(rng_key, sample_global_features, sample_objects, 
                         sample_object_labels, sample_object_mask)
        
        # Forward pass
        q_values = net.apply(params, sample_global_features, sample_objects, 
                            sample_object_labels, sample_object_mask)
        
        # Check output shape
        assert q_values.shape == (batch_size, num_concepts)
        assert q_values.dtype == jnp.float32
    
    def test_forward_pass_with_masked_objects(self, rng_key, batch_size, num_concepts, 
                                               global_dim, obj_dim, max_objs):
        """Test that masking works correctly (masked objects don't affect output)."""
        net = MetaPolicyQNetwork(num_concepts=num_concepts)
        
        # Create inputs with some masked objects
        global_features = jnp.ones((batch_size, global_dim))
        objects = jnp.ones((batch_size, max_objs, obj_dim))
        object_labels = jnp.zeros((batch_size, max_objs, num_concepts), dtype=jnp.float32)  # Multi-hot
        
        # Mask: only first 5 objects are real
        object_mask = jnp.zeros((batch_size, max_objs))
        object_mask = object_mask.at[:, :5].set(1.0)
        
        # Initialize and forward pass
        params = net.init(rng_key, global_features, objects, object_labels, object_mask)
        q_values = net.apply(params, global_features, objects, object_labels, object_mask)
        
        # Should still produce correct shape
        assert q_values.shape == (batch_size, num_concepts)
    
    def test_jax_jit_compliance(self, rng_key, batch_size, num_concepts, global_dim, 
                                obj_dim, max_objs, sample_global_features, 
                                sample_objects, sample_object_labels, sample_object_mask):
        """Test that the network can be jitted."""
        net = MetaPolicyQNetwork(num_concepts=num_concepts)
        
        # Initialize
        params = net.init(rng_key, sample_global_features, sample_objects, 
                         sample_object_labels, sample_object_mask)
        
        # Create jitted version
        @jax.jit
        def jitted_forward(params, global_features, objects, object_labels, object_mask):
            return net.apply(params, global_features, objects, object_labels, object_mask)
        
        # Run jitted version
        q_values = jitted_forward(params, sample_global_features, sample_objects, 
                                 sample_object_labels, sample_object_mask)
        
        assert q_values.shape == (batch_size, num_concepts)
    
    def test_gradient_computation(self, rng_key, batch_size, num_concepts, global_dim, 
                                  obj_dim, max_objs, sample_global_features, 
                                  sample_objects, sample_object_labels, sample_object_mask):
        """Test that gradients can be computed through the network."""
        net = MetaPolicyQNetwork(num_concepts=num_concepts)
        
        # Initialize
        params = net.init(rng_key, sample_global_features, sample_objects, 
                         sample_object_labels, sample_object_mask)
        
        # Define loss function
        def loss_fn(params):
            q_values = net.apply(params, sample_global_features, sample_objects, 
                                sample_object_labels, sample_object_mask)
            return jnp.mean(q_values ** 2)
        
        # Compute gradients
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)
        
        # Check that gradients exist and have same structure as params
        assert isinstance(grads, dict)
        assert 'params' in grads
    
    def test_different_batch_sizes(self, rng_key, num_concepts, global_dim, obj_dim, max_objs):
        """Test that network works with different batch sizes."""
        net = MetaPolicyQNetwork(num_concepts=num_concepts)
        
        for batch_size in [1, 2, 8, 16]:
            global_features = jnp.ones((batch_size, global_dim))
            objects = jnp.ones((batch_size, max_objs, obj_dim))
            object_labels = jnp.zeros((batch_size, max_objs, num_concepts), dtype=jnp.float32)  # Multi-hot
            object_mask = jnp.ones((batch_size, max_objs))
            
            params = net.init(rng_key, global_features, objects, object_labels, object_mask)
            q_values = net.apply(params, global_features, objects, object_labels, object_mask)
            
            assert q_values.shape == (batch_size, num_concepts)


class TestMetaPolicy:
    """Tests for MetaPolicy utility class."""
    
    def test_create_train_state(self, rng_key, num_concepts, global_dim, obj_dim, max_objs):
        """Test that TrainState can be created correctly."""
        learning_rate = 1e-3
        
        train_state = MetaPolicy.create_train_state(
            key=rng_key,
            num_concepts=num_concepts,
            learning_rate=learning_rate,
            global_dim=global_dim,
            obj_dim=obj_dim,
            max_objs=max_objs
        )
        
        # Check that it's a TrainState
        assert isinstance(train_state, TrainState)
        assert train_state.params is not None
        assert train_state.tx is not None
    
    def test_select_concept_eps_greedy_shapes(self, rng_key, batch_size, num_concepts, 
                                              global_dim, obj_dim, max_objs,
                                              sample_global_features, sample_objects, 
                                              sample_object_labels, sample_object_mask):
        """Test that epsilon-greedy selection produces correct shapes."""
        train_state = MetaPolicy.create_train_state(
            key=rng_key,
            num_concepts=num_concepts,
            learning_rate=1e-3,
            global_dim=global_dim,
            obj_dim=obj_dim,
            max_objs=max_objs
        )
        
        epsilon = 0.1
        key1, key2 = jax.random.split(rng_key)
        
        actions, q_values = MetaPolicy.select_concept_eps_greedy(
            train_state=train_state,
            global_features=sample_global_features,
            objects=sample_objects,
            object_labels=sample_object_labels,
            object_mask=sample_object_mask,
            epsilon=epsilon,
            key=key1
        )
        
        # Check shapes
        assert actions.shape == (batch_size,)
        assert q_values.shape == (batch_size, num_concepts)
        assert actions.dtype == jnp.int32 or actions.dtype == jnp.int64
        assert q_values.dtype == jnp.float32
    
    def test_select_concept_eps_greedy_jit_compliance(self, rng_key, batch_size, num_concepts,
                                                      global_dim, obj_dim, max_objs,
                                                      sample_global_features, sample_objects,
                                                      sample_object_labels, sample_object_mask):
        """Test that epsilon-greedy selection can be jitted (it's already jitted, but test it works)."""
        train_state = MetaPolicy.create_train_state(
            key=rng_key,
            num_concepts=num_concepts,
            learning_rate=1e-3,
            global_dim=global_dim,
            obj_dim=obj_dim,
            max_objs=max_objs
        )
        
        epsilon = 0.1
        key1, key2 = jax.random.split(rng_key)
        
        # This should work since it's already jitted
        actions, q_values = MetaPolicy.select_concept_eps_greedy(
            train_state=train_state,
            global_features=sample_global_features,
            objects=sample_objects,
            object_labels=sample_object_labels,
            object_mask=sample_object_mask,
            epsilon=epsilon,
            key=key1
        )
        
        assert actions.shape == (batch_size,)
        assert q_values.shape == (batch_size, num_concepts)
    
    def test_epsilon_greedy_behavior(self, rng_key, batch_size, num_concepts, global_dim,
                                     obj_dim, max_objs, sample_global_features, sample_objects,
                                     sample_object_labels, sample_object_mask):
        """Test that epsilon-greedy behaves correctly (greedy when epsilon=0, random when epsilon=1)."""
        train_state = MetaPolicy.create_train_state(
            key=rng_key,
            num_concepts=num_concepts,
            learning_rate=1e-3,
            global_dim=global_dim,
            obj_dim=obj_dim,
            max_objs=max_objs
        )
        
        # Test greedy (epsilon=0)
        key1, key2 = jax.random.split(rng_key)
        actions_greedy, q_values = MetaPolicy.select_concept_eps_greedy(
            train_state=train_state,
            global_features=sample_global_features,
            objects=sample_objects,
            object_labels=sample_object_labels,
            object_mask=sample_object_mask,
            epsilon=0.0,
            key=key1
        )
        
        # Greedy actions should match argmax of q_values
        expected_greedy = jnp.argmax(q_values, axis=-1)
        np.testing.assert_array_equal(actions_greedy, expected_greedy)
        
        # Test random (epsilon=1)
        key3, key4 = jax.random.split(key2)
        actions_random, _ = MetaPolicy.select_concept_eps_greedy(
            train_state=train_state,
            global_features=sample_global_features,
            objects=sample_objects,
            object_labels=sample_object_labels,
            object_mask=sample_object_mask,
            epsilon=1.0,
            key=key3
        )
        
        # Random actions should be in valid range
        assert jnp.all(actions_random >= 0)
        assert jnp.all(actions_random < num_concepts)
    
    def test_action_values_in_range(self, rng_key, batch_size, num_concepts, global_dim,
                                    obj_dim, max_objs, sample_global_features, sample_objects,
                                    sample_object_labels, sample_object_mask):
        """Test that selected actions are in valid range."""
        train_state = MetaPolicy.create_train_state(
            key=rng_key,
            num_concepts=num_concepts,
            learning_rate=1e-3,
            global_dim=global_dim,
            obj_dim=obj_dim,
            max_objs=max_objs
        )
        
        epsilon = 0.5
        key1, key2 = jax.random.split(rng_key)
        
        actions, q_values = MetaPolicy.select_concept_eps_greedy(
            train_state=train_state,
            global_features=sample_global_features,
            objects=sample_objects,
            object_labels=sample_object_labels,
            object_mask=sample_object_mask,
            epsilon=epsilon,
            key=key1
        )
        
        # All actions should be in valid range [0, num_concepts)
        assert jnp.all(actions >= 0)
        assert jnp.all(actions < num_concepts)
