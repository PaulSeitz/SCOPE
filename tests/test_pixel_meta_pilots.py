"""Input isolation, candidate identity, gradients, and pilot config contracts."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path
import yaml

from concept_based_learning.environments.datatypes import LabeledObjectObservation
from concept_based_learning.networks.meta_factory import create_meta_network
from concept_based_learning.networks.pixel_meta_network import (
    candidate_marked_stacks, pixel_stack_channels,
)


def config(arch, chunk=1):
    return dict(META_ARCHITECTURE=arch, META_INPUT_MODE="pixel_embed",
                MAX_OBJECTS=3, META_USE_NOOP_TARGET=True,
                META_HIDDEN_SIZE=8, META_FLAT_NUM_LAYERS=1, META_EMBED_DIM=8,
                META_PIXEL_CNN_CHANNELS=[2], META_PIXEL_CNN_KERNEL_SIZE=[3, 3],
                META_CANDIDATE_CHUNK_SIZE=chunk)


def objects():
    def field(values):
        return jnp.tile(jnp.array(values)[None], (2, 1))
    zero=field([0, 0, 0])
    return LabeledObjectObservation(
        x=field([16, 80, 0]), y=field([20, 100, 0]),
        width=field([20, 20, 0]), height=field([20, 20, 0]),
        active=field([1, 1, 0]), semantic_label=field([1, 2, 0]),
        visual_id=zero, state=zero, orientation=zero,
        can_go_offscreen=zero, was_killed=zero, was_collected=zero,
    )


def image():
    return jnp.arange(2*12*12, dtype=jnp.float32).reshape(2, 12, 12, 1) % 256


def test_stack_conversion_preserves_temporal_and_spatial_order():
    x=jnp.arange(2*3*4).reshape(2,3,4,1).astype(jnp.float32)/100
    converted=pixel_stack_channels(x)
    np.testing.assert_array_equal(converted[...,0],x[0,...,0])
    np.testing.assert_array_equal(converted[...,1],x[1,...,0])


@pytest.mark.parametrize("arch",["pixel_slots","pixel_candidates"])
def test_excluded_annotations_and_raw_state_cannot_change_scores(arch):
    net=create_meta_network(config(arch));obs=objects();pixels=image()
    params=net.init(jax.random.PRNGKey(0),None,obs,False,pixel_obs=pixels)
    q=net.apply(params,None,obs,False,pixel_obs=pixels)
    changed=obs.replace(**{name:jnp.ones_like(getattr(obs,name))*7 for name in (
        "semantic_label","visual_id","state","orientation",
        "can_go_offscreen","was_killed","was_collected")})
    alternative=net.apply(params,{"secret_state":jnp.array(999.)},changed,
                          False,pixel_obs=pixels)
    assert q.shape==(3,)
    np.testing.assert_array_equal(q,alternative)
    different_pixels=net.apply(params,None,obs,False,pixel_obs=jnp.flip(pixels,axis=1))
    assert not np.allclose(q,different_pixels)
    if arch=="pixel_slots":
        np.testing.assert_array_equal(q,net.apply(params,None,None,False,pixel_obs=pixels))


def test_candidate_permutation_swaps_scores_but_keeps_noop():
    net=create_meta_network(config("pixel_candidates"));obs=objects();pixels=image()
    params=net.init(jax.random.PRNGKey(1),None,obs,False,pixel_obs=pixels)
    q=net.apply(params,None,obs,False,pixel_obs=pixels)
    swapped=jax.tree.map(lambda x:x[:,jnp.array([1,0,2])],obs)
    q_swapped=net.apply(params,None,swapped,False,pixel_obs=pixels)
    np.testing.assert_allclose(q_swapped,q[jnp.array([1,0,2])],atol=1e-6)
    marked=candidate_marked_stacks(pixels,obs,action_dim=3,noop_idx=2)
    assert not np.array_equal(marked[0],marked[1])
    assert not np.array_equal(marked[2],pixels/255.)


def test_chunking_preserves_scores_and_gradients():
    obs=objects();pixels=image()
    first=create_meta_network(config("pixel_candidates",1))
    second=create_meta_network(config("pixel_candidates",2))
    params=first.init(jax.random.PRNGKey(2),None,obs,False,pixel_obs=pixels)
    def loss(net,p):
        return jnp.square(net.apply(p,None,obs,True,pixel_obs=pixels)-1).mean()
    value_a,grad_a=jax.value_and_grad(lambda p:loss(first,p))(params)
    value_b,grad_b=jax.value_and_grad(lambda p:loss(second,p))(params)
    np.testing.assert_allclose(value_a,value_b,rtol=1e-5,atol=1e-6)
    for a,b in zip(jax.tree.leaves(grad_a),jax.tree.leaves(grad_b)):
        assert np.all(np.isfinite(a))
        np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-5)
    assert any(np.any(x!=0) for x in jax.tree.leaves(grad_a))


def test_raw_slot_network_has_finite_training_gradients():
    net=create_meta_network(config("pixel_slots"));pixels=image()
    params=net.init(jax.random.PRNGKey(5),None,None,False,pixel_obs=pixels)
    grads=jax.grad(lambda p:jnp.square(net.apply(p,None,None,True,pixel_obs=pixels)-1).mean())(params)
    assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(grads))
    assert any(np.any(x!=0) for x in jax.tree.leaves(grads))


def test_pure_pixel_modes_reject_hidden_auxiliary_inputs():
    with pytest.raises(ValueError,match="target-lock"):
        create_meta_network({**config("pixel_slots"),"META_TARGET_LOCKED_INPUT":True})


@pytest.mark.parametrize('env',['seaquest','namethisgame'])
def test_pilot_configs_match_budget_rewards_and_isolate_eval(env):
    root=Path(__file__).resolve().parents[1]
    base=yaml.safe_load((root/f'configs/experiment/paper_eval/thesis_final/tf_{env}_tandem_pixel_seed44.yaml').read_text())
    base={**base,**base['alg']}
    for variant in ('slots','candidates'):
        stem=f'pm_{env}_{variant}_seed44'
        directory=root/'configs/experiment/paper_eval/meta_selector_pilots'
        raw=yaml.safe_load((directory/f'{stem}.yaml').read_text())
        cfg={**raw,**raw['alg']}
        for key in ('NUM_ENVS','NUM_STEPS','TOTAL_TIMESTEPS','TOTAL_TIMESTEPS_DECAY',
                    'NUM_MINIBATCHES','NUM_EPOCHS','GAMMA','LAMBDA','SKILL_LAMBDA',
                    'META_SWITCH_PENALTY','META_REWARD_LAMBDA','SKILL_INPUT_MODE',
                    'SKILL_DEATH_PENALIZE_COLLECT','DESTROY_CONTACT_FAIL',
                    'COLLECT_SPARSE_ONLY','AVOID_SPARSE_ONLY','AVOID_KILLED_EVENT_MODE',
                    'OFF_POLICY','RELABEL_ALL_OBJECTS','SEED'):
            assert cfg[key]==base[key],key
        evaluate=yaml.safe_load((directory/f'{stem}_eval.yaml').read_text())
        assert evaluate['PAPER_EVAL_MOD_KEYS']==['base']
        assert evaluate['PAPER_EVAL_ARCHITECTURE_OF']==stem
        assert stem in evaluate['PAPER_EVAL_META_CKPT']
        assert stem in evaluate['PAPER_EVAL_SKILLS_CKPT']
        assert cfg['META_ARCHITECTURE']==f'pixel_{variant}'
        assert cfg['META_TARGET_LOCKED_INPUT'] is False
