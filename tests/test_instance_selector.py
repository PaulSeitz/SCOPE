"""
Tests for instance_selector.py - InstanceSelector.

Tests cover:
- Network initialization and forward pass
- Shape correctness
- JAX compliance (jitting, gradients)
- Attention mechanism behavior
- Target object selection
- Masking behavior
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
import optax

from src.agent.components.instance_selector import InstanceSelector


class TestInstanceSelector:
    """Tests for InstanceSelector module."""
    
    def test_network_initialization(self, rng_key, num_concepts, global_dim, obj_dim, max_objs):
        """Test that the network can be initialized with correct parameters."""
        net = InstanceSelector(num_concepts=num_concepts)
        
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
        net = InstanceSelector(num_concepts=num_concepts)
        
        # Initialize
        params = net.init(rng_key, sample_global_features, sample_objects, sample_object_labels, sample_object_mask)
        
        # Forward pass
        attn_weights = net.apply(params, sample_global_features, sample_objects, sample_object_labels, sample_object_mask)
        
        # Check output shape: [Batch, Num_Concepts, Max_Objs]
        assert attn_weights.shape == (batch_size, num_concepts, max_objs)
        assert attn_weights.dtype == jnp.float32
    
    def test_attention_weights_sum_to_one(self, rng_key, batch_size, num_concepts, global_dim,
                                          obj_dim, max_objs, sample_global_features,
                                          sample_objects, sample_object_labels, sample_object_mask):
        """Test that attention weights sum to 1 for each concept (probability distribution)."""
        net = InstanceSelector(num_concepts=num_concepts)
        
        params = net.init(rng_key, sample_global_features, sample_objects, sample_object_labels, sample_object_mask)
        attn_weights = net.apply(params, sample_global_features, sample_objects, sample_object_labels, sample_object_mask)
        
        # Sum over object dimension should be 1 for each concept
        sums = jnp.sum(attn_weights, axis=-1)  # [Batch, Num_Concepts]
        
        # Should be approximately 1.0 (allowing for floating point errors)
        np.testing.assert_array_almost_equal(sums, jnp.ones((batch_size, num_concepts)), decimal=5)
    
    def test_attention_weights_non_negative(self, rng_key, batch_size, num_concepts, global_dim,
                                            obj_dim, max_objs, sample_global_features,
                                            sample_objects, sample_object_labels, sample_object_mask):
        """Test that attention weights are non-negative (softmax output)."""
        net = InstanceSelector(num_concepts=num_concepts)
        
        params = net.init(rng_key, sample_global_features, sample_objects, sample_object_labels, sample_object_mask)
        attn_weights = net.apply(params, sample_global_features, sample_objects, sample_object_labels, sample_object_mask)
        
        # All weights should be >= 0
        assert jnp.all(attn_weights >= 0)
    
    def test_masking_behavior(self, rng_key, batch_size, num_concepts, global_dim,
                              obj_dim, max_objs):
        """Test that masked objects receive zero attention."""
        net = InstanceSelector(num_concepts=num_concepts)
        
        # Create inputs with some masked objects
        global_features = jnp.ones((batch_size, global_dim))
        objects = jnp.ones((batch_size, max_objs, obj_dim))
        object_labels = jnp.zeros((batch_size, max_objs, num_concepts), dtype=jnp.float32)  # Multi-hot
        
        # Mask: only first 5 objects are real
        object_mask = jnp.zeros((batch_size, max_objs))
        object_mask = object_mask.at[:, :5].set(1.0)
        
        params = net.init(rng_key, global_features, objects, object_labels, object_mask)
        attn_weights = net.apply(params, global_features, objects, object_labels, object_mask)
        
        # Masked objects (indices 5+) should have zero attention
        masked_weights = attn_weights[:, :, 5:]
        np.testing.assert_array_almost_equal(masked_weights, jnp.zeros_like(masked_weights), decimal=5)
    
    def test_jax_jit_compliance(self, rng_key, batch_size, num_concepts, global_dim,
                                obj_dim, max_objs, sample_global_features,
                                sample_objects, sample_object_labels, sample_object_mask):
        """Test that the network can be jitted."""
        net = InstanceSelector(num_concepts=num_concepts)
        
        # Initialize
        params = net.init(rng_key, sample_global_features, sample_objects, sample_object_labels, sample_object_mask)
        
        # Create jitted version
        @jax.jit
        def jitted_forward(params, global_features, objects, object_labels, object_mask):
            return net.apply(params, global_features, objects, object_labels, object_mask)
        
        # Run jitted version
        attn_weights = jitted_forward(params, sample_global_features, sample_objects, sample_object_labels, sample_object_mask)
        
        assert attn_weights.shape == (batch_size, num_concepts, max_objs)
    
    def test_gradient_computation(self, rng_key, batch_size, num_concepts, global_dim,
                                  obj_dim, max_objs, sample_global_features,
                                  sample_objects, sample_object_labels, sample_object_mask):
        """Test that gradients can be computed through the network."""
        net = InstanceSelector(num_concepts=num_concepts)
        
        # Initialize
        params = net.init(rng_key, sample_global_features, sample_objects, sample_object_labels, sample_object_mask)
        
        # Define loss function
        def loss_fn(params):
            attn_weights = net.apply(params, sample_global_features, sample_objects, sample_object_labels, sample_object_mask)
            return jnp.mean(attn_weights ** 2)
        
        # Compute gradients
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)
        
        # Check that gradients exist and have same structure as params
        assert isinstance(grads, dict)
        assert 'params' in grads
    
    def test_different_batch_sizes(self, rng_key, num_concepts, global_dim, obj_dim, max_objs):
        """Test that network works with different batch sizes."""
        net = InstanceSelector(num_concepts=num_concepts)
        
        for batch_size in [1, 2, 8, 16]:
            global_features = jnp.ones((batch_size, global_dim))
            objects = jnp.ones((batch_size, max_objs, obj_dim))
            object_labels = jnp.zeros((batch_size, max_objs, num_concepts), dtype=jnp.float32)  # Multi-hot
            object_mask = jnp.ones((batch_size, max_objs))
            
            params = net.init(rng_key, global_features, objects, object_labels, object_mask)
            attn_weights = net.apply(params, global_features, objects, object_labels, object_mask)
            
            assert attn_weights.shape == (batch_size, num_concepts, max_objs)
    
    def test_get_target_object_shapes(self, batch_size, max_objs, obj_dim, num_concepts):
        """Test that get_target_object produces correct shapes."""
        # Create dummy attention weights
        attn_weights = jnp.ones((batch_size, num_concepts, max_objs)) / max_objs
        
        # Create dummy objects
        objects = jnp.ones((batch_size, max_objs, obj_dim))
        
        # Create concept IDs (one per batch item)
        concept_id = jnp.array([0, 1, 2, 3])[:batch_size]  # Each batch item selects a different concept
        
        target_obj, target_idx = InstanceSelector.get_target_object(objects, attn_weights, concept_id)
        
        # Check shapes
        assert target_obj.shape == (batch_size, obj_dim)
        assert target_idx.shape == (batch_size,)
        assert target_idx.dtype == jnp.int32 or target_idx.dtype == jnp.int64
    
    def test_get_target_object_jit_compliance(self, batch_size, max_objs, obj_dim, num_concepts):
        """Test that get_target_object can be jitted."""
        # Create dummy inputs
        attn_weights = jnp.ones((batch_size, num_concepts, max_objs)) / max_objs
        objects = jnp.ones((batch_size, max_objs, obj_dim))
        concept_id = jnp.array([0, 1, 2, 3])[:batch_size]
        
        # Create jitted version
        @jax.jit
        def jitted_get_target(objects, attn_weights, concept_id):
            return InstanceSelector.get_target_object(objects, attn_weights, concept_id)
        
        # Run jitted version
        target_obj, target_idx = jitted_get_target(objects, attn_weights, concept_id)
        
        assert target_obj.shape == (batch_size, obj_dim)
        assert target_idx.shape == (batch_size,)
    
    def test_get_target_object_selection(self, batch_size, max_objs, obj_dim, num_concepts):
        """Test that get_target_object correctly selects objects based on attention."""
        # Create objects with distinct values
        objects = jnp.zeros((batch_size, max_objs, obj_dim))
        # Set first object to have value 1.0, others to 0.0
        objects = objects.at[:, 0, :].set(1.0)
        
        # Create attention weights that focus on first object for all concepts
        attn_weights = jnp.zeros((batch_size, num_concepts, max_objs))
        attn_weights = attn_weights.at[:, :, 0].set(1.0)  # All attention on first object
        
        # Select concept 0 for all batch items
        concept_id = jnp.zeros((batch_size,), dtype=jnp.int32)
        
        target_obj, target_idx = InstanceSelector.get_target_object(objects, attn_weights, concept_id)
        
        # Should select first object (index 0)
        np.testing.assert_array_equal(target_idx, jnp.zeros((batch_size,), dtype=jnp.int32))
        # Should have value 1.0
        np.testing.assert_array_almost_equal(target_obj, jnp.ones((batch_size, obj_dim)), decimal=5)
    
    def test_get_target_object_different_concepts(self, batch_size, max_objs, obj_dim, num_concepts):
        """Test that different concepts can select different objects."""
        # Create objects with distinct values
        objects = jnp.zeros((batch_size, max_objs, obj_dim))
        for i in range(min(batch_size, max_objs)):
            objects = objects.at[:, i, :].set(float(i + 1))
        
        # Create attention weights: each concept focuses on a different object
        attn_weights = jnp.zeros((batch_size, num_concepts, max_objs))
        for c in range(min(num_concepts, max_objs)):
            attn_weights = attn_weights.at[:, c, c].set(1.0)
        
        # Each batch item selects a different concept
        concept_id = jnp.arange(batch_size, dtype=jnp.int32) % num_concepts
        
        target_obj, target_idx = InstanceSelector.get_target_object(objects, attn_weights, concept_id)
        
        # Check that indices are correct
        expected_indices = jnp.arange(batch_size, dtype=jnp.int32) % num_concepts
        np.testing.assert_array_equal(target_idx, expected_indices)
    
    def test_attention_mechanism_query_key_value(self, rng_key, batch_size, num_concepts, global_dim,
                                                 obj_dim, max_objs):
        """Test that the attention mechanism properly uses queries, keys, and values."""
        net = InstanceSelector(num_concepts=num_concepts, embed_dim=64)
        
        # Create inputs with known structure
        global_features = jnp.ones((batch_size, global_dim)) * 0.5
        objects = jnp.ones((batch_size, max_objs, obj_dim)) * 0.3
        object_labels = jnp.zeros((batch_size, max_objs, num_concepts), dtype=jnp.float32)  # Multi-hot
        object_mask = jnp.ones((batch_size, max_objs))
        
        params = net.init(rng_key, global_features, objects, object_labels, object_mask)
        attn_weights = net.apply(params, global_features, objects, object_labels, object_mask)
        
        # Should produce valid attention weights
        assert attn_weights.shape == (batch_size, num_concepts, max_objs)
        assert jnp.all(attn_weights >= 0)
        assert jnp.all(attn_weights <= 1)
        
        # Sum should be 1 for each concept
        sums = jnp.sum(attn_weights, axis=-1)
        np.testing.assert_array_almost_equal(sums, jnp.ones((batch_size, num_concepts)), decimal=5)
