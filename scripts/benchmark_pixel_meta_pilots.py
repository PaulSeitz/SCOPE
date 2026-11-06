#!/usr/bin/env python3
"""Synthetic selector forward/backward benchmark, NOT an RL learning result.

Use the target GPU. Batch size should be increased towards NUM_ENVS for rollout
inference and towards NUM_ENVS*NUM_STEPS/NUM_MINIBATCHES for meta gradients.
Each batch element contains all MAX_OBJECTS slots, including inactive slots.
"""
from pathlib import Path
import argparse
import json
import statistics
import time
import jax
import jax.numpy as jnp
import yaml
from concept_based_learning.environments.datatypes import LabeledObjectObservation
from concept_based_learning.networks.meta_factory import create_meta_network


def run(path,batch_size,repetitions):
    cfg=yaml.safe_load(path.read_text());cfg={**cfg,**cfg.get('alg',{})}
    n=cfg['MAX_OBJECTS'];t=cfg['FRAME_STACK_SIZE']
    h,w=cfg.get('META_PIXEL_RESIZE_SHAPE',[84,84])
    pixels=jax.random.randint(jax.random.PRNGKey(0),(batch_size,t,h,w,1),0,256,dtype=jnp.uint8)
    def field(a):
        return jnp.broadcast_to(jnp.asarray(a),(batch_size,t,n))
    z=field(jnp.zeros(n));active=field(jnp.ones(n)).at[:,:,-1].set(0)
    obs=LabeledObjectObservation(
        x=field(jnp.arange(n)*3%140),y=field(jnp.arange(n)*5%180),
        width=field(jnp.ones(n)*8),height=field(jnp.ones(n)*8),active=active,
        semantic_label=z,visual_id=z,state=z,orientation=z,can_go_offscreen=z,
        was_killed=z,was_collected=z)
    model=create_meta_network(cfg)
    single=jax.tree.map(lambda x:x[0],obs)
    start=time.perf_counter()
    params=model.init(jax.random.PRNGKey(1),None,single,False,pixel_obs=pixels[0])
    def scores(p,images,objects):
        return jax.vmap(lambda im,ob:model.apply(p,None,ob,False,pixel_obs=im))(images,objects)
    forward=jax.jit(scores)
    update=jax.jit(jax.value_and_grad(
        lambda p,im,ob:jnp.square(scores(p,im,ob)[:,0]-1).mean()))
    jax.block_until_ready(forward(params,pixels,obs))
    jax.block_until_ready(update(params,pixels,obs))
    compile_seconds=time.perf_counter()-start
    def measure(fn):
        values=[]
        for _ in range(repetitions):
            begin=time.perf_counter();jax.block_until_ready(fn(params,pixels,obs))
            values.append(time.perf_counter()-begin)
        return statistics.median(values)
    result=dict(config=path.stem,device=str(jax.devices()[0]),batch_size=batch_size,
                candidates=n,parameters=sum(x.size for x in jax.tree.leaves(params)),
                init_compile_seconds=compile_seconds,
                median_forward_seconds=measure(forward),
                median_loss_gradient_seconds=measure(update),
                memory_stats=jax.devices()[0].memory_stats())
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.batch_size<1 or args.repetitions<1:parser.error('Counts must be positive.')
    result=run(args.config,args.batch_size,args.repetitions)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
