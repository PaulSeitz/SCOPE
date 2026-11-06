#!/usr/bin/env python3
"""Print a raw-slot train/home-eval/transfer chain; execute only with --run."""
from pathlib import Path
import argparse
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
TARGETS=('seaquest','namethisgame','spaceinvaders','beamrider','asteroids',
         'pong','breakout','freeway','phoenix')


def stems(env,seed,reuse_pilot=False):
    if reuse_pilot:
        if (env,seed)!=('seaquest',44):
            raise ValueError('Pilot reuse is available only for Seaquest seed 44.')
        return [('ps_seaquest_slots_seed44_full_eval','eval')]+[
            (f'ps_zs_slots_seaquest_to_{target}_seed44','eval')
            for target in TARGETS if target!=env]
    source=f'tf_{env}_tandem_pixel_slots_seed{seed}'
    return [(source,'train'),(source+'_eval','eval')]+[
        (f'tf_zs_tandem_pixel_slots_skills_{env}_to_{target}_seed{seed}','eval')
        for target in TARGETS if target!=env]


def verify_pilot(root):
    directory=root/'outputs/models/paper_eval/meta_selector_pilots/pm_seaquest_slots_seed44/seaquest'
    prefix='hierarchical_pqn_seaquest_seed44'
    for suffix in ('_config.yaml','_skills.safetensors','_meta.safetensors'):
        if not (directory/(prefix+suffix)).is_file():
            raise RuntimeError(f'Missing pulled pilot artifact: {directory/(prefix+suffix)}')
    actual=yaml.safe_load((directory/(prefix+'_config.yaml')).read_text())
    actual={**actual,**actual.get('alg',{})}
    expected=yaml.safe_load((root/'configs/experiment/paper_eval/pixel_meta_final/tf_seaquest_tandem_pixel_slots_seed44.yaml').read_text())
    expected={**expected,**expected['alg']}
    keys=('ENV_NAME','META_ARCHITECTURE','META_INPUT_MODE','META_HIDDEN_SIZE',
          'META_EMBED_DIM','META_FLAT_NUM_LAYERS','META_PIXEL_CNN_CHANNELS',
          'NUM_SKILLS','SKILL_NAMES','SKILL_INPUT_MODE','HIDDEN_SIZE','NUM_LAYERS',
          'NUM_ENVS','NUM_STEPS','TOTAL_TIMESTEPS','GAMMA','LAMBDA','SKILL_LAMBDA',
          'LR','SKILL_LR','META_LR','NUM_EPOCHS','NUM_MINIBATCHES','FRAME_SKIP',
          'FRAME_STACK_SIZE','USE_POV_FACING_FRAME','USE_POV_RELATIVE_MAPPER',
          'EGO_PSI_ANGLE_BINS','SKILL_DEATH_PENALIZE_COLLECT','DESTROY_CONTACT_FAIL',
          'COLLECT_SPARSE_ONLY','AVOID_SPARSE_ONLY','AVOID_KILLED_EVENT_MODE',
          'COLLECT_EARLY_WHEN_NO_THREATS','META_SWITCH_PENALTY','OFF_POLICY',
          'RELABEL_ALL_OBJECTS','SEED')
    differences={k:(actual.get(k),expected.get(k)) for k in keys
                 if actual.get(k)!=expected.get(k)}
    if differences:raise RuntimeError(f'Pilot/final protocol mismatch: {differences}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env',choices=['seaquest','namethisgame'],required=True)
    parser.add_argument('--seed',type=int,choices=[44,45,46],required=True)
    parser.add_argument('--reuse-pilot',action='store_true')
    parser.add_argument('--run',action='store_true',help='Execute; default prints only.')
    args=parser.parse_args()
    chain=stems(args.env,args.seed,args.reuse_pilot)
    if args.run and args.reuse_pilot:verify_pilot(ROOT)
    for stem,phase in chain:
        command=[sys.executable,str(ROOT/'scripts/run_paper_eval.py'),
                 '--only',stem,'--phase',phase,'--resume','--abort-on-failure']
        print(' '.join(command),flush=True)
        if args.run:subprocess.run(command,cwd=ROOT,check=True)
