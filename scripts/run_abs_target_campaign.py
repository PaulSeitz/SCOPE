#!/usr/bin/env python3
"""Detached absolute-target train/evaluation campaign with per-job status and logs."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODS = {
    'seaquest': ['base', 'shift_lanes', 'random_spawns', 'gravity', 'faster_enemies', 'no_enemies', 'change_background_color'],
    'namethisgame': ['base', 'shift_tentacle_spawn_x', 'slower_player', 'faster_shark', 'no_shark', 'no_shark_short_tentacles', 'change_background_color'],
}

def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def stem(game, seed):
    return f'tf_{game}_skills_ego_abstarget_seed{seed}'


def train_cfg(game, seed):
    import yaml
    root = yaml.safe_load((ROOT/'configs/config.yaml').read_text())
    root.pop('defaults', None)
    raw = yaml.safe_load((ROOT/f'configs/experiment/paper_eval/abs_target_ablation/{stem(game, seed)}.yaml').read_text())
    cfg = {**root, **{k:v for k,v in raw.items() if k != 'alg'}, **raw['alg']}
    cfg['SAVE_PATH'] = str(ROOT/Path(cfg['SAVE_PATH']))
    cfg['WANDB_MODE'] = 'offline'
    cfg['RTPT_ENABLED'] = True
    cfg['RTPT_EXPERIMENT_NAME'] = f'abs_{game[:3]}_{seed}'
    cfg['RECORD_VIDEO'] = False
    cfg['VIDEO_AT_END'] = False
    assert cfg['SKILL_INPUT_MODE'] == 'ego_abs_target'
    assert cfg['TOTAL_TIMESTEPS'] == 50000000
    return cfg


def checkpoint(game, seed):
    return ROOT/f'outputs/models/paper_eval/abs_target_ablation/{stem(game,seed)}/{game}/hierarchical_pqn_{game}_seed{seed}.safetensors'


def worker(args):
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import yaml
    # Match the existing frozen-evaluation worker's host-side action tables.
    import inspect
    from importlib import import_module
    from jaxtari.core import GAME_MODULES
    from jaxtari.environment import JaxEnvironment
    module = import_module(GAME_MODULES[args.game])
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls is not JaxEnvironment and issubclass(cls, JaxEnvironment) and hasattr(cls, 'ACTION_SET'):
            cls.ACTION_SET = np.asarray(cls.ACTION_SET, dtype=np.int32)
    cfg = train_cfg(args.game, args.seed)
    if args.mode == 'train':
        from main import single_run
        single_run(cfg)
        path = checkpoint(args.game, args.seed)
        assert path.is_file(), path
        from safetensors import safe_open
        with safe_open(path, framework='numpy') as model:
            kernels = [model.get_tensor(k) for k in model.keys() if k.endswith('Dense_0,kernel')]
            assert kernels and all(k.shape[-2:] == (36,128) for k in kernels)
            assert all(np.isfinite(model.get_tensor(k)).all() for k in model.keys())
        write_json(ROOT/'status'/f'{stem(args.game,args.seed)}.trained.json', {
            'checkpoint':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'completed':dt.datetime.now().astimezone().isoformat(), 'requested_steps':50000000,
            'updates':50000000//(1024*32), 'input_mode':'ego_abs_target',
        })
    else:
        from concept_based_learning.paper_eval.eval_harness import run_mod_eval
        cfg.update(WANDB_MODE='disabled', RTPT_ENABLED=False, EVAL_PARALLEL_MODS=False)
        path = ROOT/'eval_jobs'/f'{stem(args.game,args.seed)}__{args.mod}.json'
        payload = run_mod_eval(
            agent_kind='skills', config=cfg, env_name=args.game,
            skills_path=checkpoint(args.game,args.seed), mod_keys=[args.mod],
            num_episodes=3, num_seeds=3, base_seed=10000+args.seed,
            max_frames=27000, output_json=path, master_csv_dir=ROOT/'eval_jobs'/f'csv_{args.game}_{args.seed}_{args.mod}',
            wandb_mode='disabled', publish_report=False,
        )
        result = payload['results'][args.mod]
        assert len(result['returns']) == 9 and np.isfinite(result['returns']).all()
        write_json(ROOT/'status'/f'{stem(args.game,args.seed)}__{args.mod}.done.json', {'result':str(path)})


def aggregate(game, seed):
    paths=[ROOT/'eval_jobs'/f'{stem(game,seed)}__{mod}.json' for mod in MODS[game]]
    if not all((ROOT/'status'/f'{stem(game,seed)}__{mod}.done.json').exists() for mod in MODS[game]):
        return
    payloads=[json.loads(p.read_text()) for p in paths]
    merged=dict(payloads[0]); merged['results']={}
    for payload,mod in zip(payloads,MODS[game]):
        assert len(payload['results'][mod]['returns']) == 9
        merged['results'][mod]=payload['results'][mod]
    merged['num_episodes']=3; merged['num_seeds']=3
    merged['campaign']=str(ROOT)
    write_json(ROOT/'eval'/f'{stem(game,seed)}_eval_mods.json',merged)


def external_gpu_processes(gpu, allowed):
    query = subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'], text=True)
    return [line for line in query.splitlines() if line and int(line.split(',')[0]) not in allowed and 'ollama' not in line.lower()]


def coordinate(args):
    placement=json.loads((ROOT/'placement.json').read_text())[args.host]
    for name in ('logs','status','eval_jobs','eval'):(ROOT/name).mkdir(exist_ok=True)
    pending=[{'game':g,'seed':s,'gpu':gpu,'mode':'train'} for gpu,g,s in placement]
    running=[]; completed=[]; failures=[]; eval_pending=[]; started=time.time()
    eval_gpus=sorted({x[0] for x in placement})
    all_cpus=sorted(os.sched_getaffinity(0)); free_cpus=set(all_cpus)
    def launch(job):
        # Keep CPU compilation bounded, even when many evaluations become ready.
        count=5 if job['mode']=='train' else 3
        if len(free_cpus)<count:return False
        allowed={r['process'].pid for r in running if r['gpu']==job['gpu']}
        foreign=external_gpu_processes(job['gpu'],allowed)
        if foreign:
            print(f"GPU {job['gpu']} occupied; waiting: {foreign}",flush=True)
            return False
        cpus=sorted(free_cpus)[:count];free_cpus.difference_update(cpus)
        name=f"{stem(job['game'],job['seed'])}"+(f"__{job['mod']}" if job['mode']=='eval' else '__train')
        log=(ROOT/'logs'/f'{name}.log').open('a')
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(job['gpu']), JAX_PLATFORMS='cuda,cpu',
                   XLA_PYTHON_CLIENT_MEM_FRACTION='0.35' if job['mode']=='train' else '0.078',
                   XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   CBL_EVAL_MOD_WORKER='1', WANDB_MODE='offline' if job['mode']=='train' else 'disabled',
                   SDL_VIDEODRIVER='dummy', MPLCONFIGDIR=str(ROOT/'.matplotlib'),
                   PYTHONPATH=f"{ROOT/'src'}:{Path.home()/'JAXAtari-seaquest-robustness/src'}:{ROOT}:{ROOT/'scripts'}",
                   XLA_FLAGS='--xla_gpu_enable_triton_gemm=false')
        cmd=['taskset','-c',','.join(map(str,cpus)),sys.executable,'-u',str(Path(__file__).resolve()),job['mode'],'--game',job['game'],'--seed',str(job['seed'])]
        if job['mode']=='eval':cmd+=['--mod',job['mod']]
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        running.append({**job,'process':proc,'cpus':cpus,'log':log,'started':time.time()})
        print(f"START {name} gpu={job['gpu']} pid={proc.pid}",flush=True)
        return True
    while pending or running or eval_pending:
        for job in list(pending):
            if launch(job):pending.remove(job);time.sleep(2)
        for r in list(running):
            code=r['process'].poll()
            if code is None:continue
            r['log'].close();free_cpus.update(r['cpus']);running.remove(r)
            event={k:v for k,v in r.items() if k not in ('process','log','cpus')}
            event.update(exit_code=code,elapsed_seconds=time.time()-r['started'])
            (completed if code==0 else failures).append(event)
            print(f"FINISH {event}",flush=True)
            if code==0 and r['mode']=='train':
                eval_pending.extend({'game':r['game'],'seed':r['seed'],'mod':m,'mode':'eval'} for m in MODS[r['game']])
            if code==0 and r['mode']=='eval':aggregate(r['game'],r['seed'])
        for gpu in eval_gpus:
            if any(j['gpu']==gpu for j in pending):continue
            if any(r['gpu']==gpu and r['mode']=='train' for r in running):continue
            limit=3 if args.host=='cluster1' or gpu>=4 else 1
            slots=limit-sum(r['gpu']==gpu for r in running)
            for job in list(eval_pending):
                if slots<=0:break
                if args.host=='cluster2' and gpu<4 and job['game']=='seaquest':continue
                if launch({**job,'gpu':gpu}):eval_pending.remove(job);slots-=1
        state={'host':args.host,'updated':dt.datetime.now().astimezone().isoformat(),
               'elapsed_seconds':time.time()-started,'pending_training':pending,'pending_evaluation':eval_pending,
               'running':[{k:v for k,v in r.items() if k not in ('process','log')}|{'pid':r['process'].pid} for r in running],
               'completed':completed,'failures':failures}
        write_json(ROOT/'status'/f'{args.host}.json',state)
        if pending or running or eval_pending:time.sleep(15)
    print('CAMPAIGN_FINISHED',flush=True)
    return int(bool(failures))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['coordinate','train','eval'])
    parser.add_argument('--host',choices=['cluster1','cluster2'])
    parser.add_argument('--game',choices=list(MODS));parser.add_argument('--seed',type=int)
    parser.add_argument('--mod')
    args=parser.parse_args()
    if args.mode=='coordinate':sys.exit(coordinate(args))
    worker(args)
