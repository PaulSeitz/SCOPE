#!/usr/bin/env python3
"""Train absolute-target skills, save/reload safetensors, evaluate shifted lanes.

Run on CPU with JAX_PLATFORMS=cpu or on an idle GPU. Writes only under --output.
The short evaluation is an integration check, not a performance measurement.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import jax
import numpy as np
import yaml
from safetensors.flax import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import outer_make_train
from concept_based_learning.training.skill_params import flatten_skill_params
from concept_based_learning.pqn_visualizer import load_params_from_file
from concept_based_learning.paper_eval.eval_harness import run_mod_eval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100352)
    parser.add_argument('--max-frames', type=int, default=256)
    args = parser.parse_args()
    if args.steps < 128:
        parser.error('--steps must be at least one 128-step update')
    args.output.mkdir(parents=True, exist_ok=False)
    path = ROOT / 'configs/experiment/paper_eval/abs_target_ablation/tf_seaquest_skills_ego_abstarget_seed44.yaml'
    raw = yaml.safe_load(path.read_text())
    alg = raw.pop('alg')
    cfg = {**raw, **alg, 'WANDB_MODE':'disabled', 'RTPT_ENABLED':False,
           'SAVE_TRAINING_METRICS_CSV':False, 'RECORD_VIDEO':False,
           'NUM_ENVS':16, 'NUM_STEPS':8, 'NUM_MINIBATCHES':4, 'NUM_EPOCHS':1,
           'TOTAL_TIMESTEPS':args.steps, 'CURRICULUM_ENABLED':False,
           'SAVE_PATH':str(args.output), 'EVAL_PARALLEL_MODS':False}
    train, _, _ = outer_make_train(cfg)
    result = jax.block_until_ready(jax.jit(train)(jax.random.PRNGKey(44), None, None))
    states = result['runner_state'][0]
    params = tuple(s.params for s in states) if isinstance(states, tuple) else states.params
    flat = flatten_skill_params(params)
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(params))
    checkpoint = args.output / 'skills.safetensors'
    save_file(flat, str(checkpoint))
    (args.output / 'skills_config.yaml').write_text(yaml.safe_dump(cfg))
    restored = flatten_skill_params(load_params_from_file(str(checkpoint)))
    assert flat.keys() == restored.keys()
    for key in flat:
        np.testing.assert_array_equal(flat[key], restored[key])
    kernels = [v for k, v in flat.items() if k.endswith('Dense_0,kernel')]
    assert kernels and all(v.shape[-2:] == (36, 128) for v in kernels)
    print('TRAIN_AND_CHECKPOINT_OK', flush=True)
    payload = run_mod_eval(agent_kind='skills', config=cfg, env_name='seaquest',
                           skills_path=checkpoint, mod_keys=['shift_lanes'],
                           num_episodes=1, num_seeds=1, max_frames=args.max_frames,
                           output_json=args.output/'shift_lanes_mods.json',
                           master_csv_dir=args.output/'csv', wandb_mode='disabled',
                           publish_report=False)
    returns = payload['results']['shift_lanes']['returns']
    assert len(returns) == 1 and np.isfinite(returns).all()
    (args.output/'smoke_passed.json').write_text(json.dumps({
        'requested_steps':args.steps, 'input_dim':36,
        'checkpoint_roundtrip':True, 'shift_lanes_returns':returns}, indent=2))
    print('SMOKE_PASSED', flush=True)


if __name__ == '__main__':
    main()
