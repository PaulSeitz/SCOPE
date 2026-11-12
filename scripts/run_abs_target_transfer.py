#!/usr/bin/env python3
"""Run the ten base-only absolute-target cross-game evaluations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BATCH=ROOT/'transfer_20261111'

def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def worker(job):
    import inspect
    from importlib import import_module
    import numpy as np
    from jaxtari.core import GAME_MODULES
    from jaxtari.environment import JaxEnvironment
    from concept_based_learning.paper_eval.eval_harness import run_mod_eval
    module=import_module(GAME_MODULES[job['target']])
    for _,cls in inspect.getmembers(module,inspect.isclass):
        if cls is not JaxEnvironment and issubclass(cls,JaxEnvironment) and hasattr(cls,'ACTION_SET'):
            cls.ACTION_SET=np.asarray(cls.ACTION_SET,dtype=np.int32)
    ckpt=Path(job['checkpoint'])
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==job['checkpoint_sha256']
    cfg=job['config'];assert cfg['ENV_NAME']==job['target'] and cfg['SKILL_INPUT_MODE']=='ego_abs_target'
    payload=run_mod_eval(agent_kind='transfer',config=cfg,env_name=job['target'],skills_path=ckpt,
        mod_keys=['base'],num_episodes=3,num_seeds=3,base_seed=10000+job['seed'],max_frames=27000,
        output_json=BATCH/'eval'/f"{job['name']}_mods.json",master_csv_dir=BATCH/'csv'/job['name'],
        wandb_mode='disabled',publish_report=False,
        extra_meta={'source_env':job['source'],'target_env':job['target'],'source_seed':job['seed'],
                    'zero_shot':True,'checkpoint_sha256':job['checkpoint_sha256']})
    stats=payload['results']['base']
    assert len(stats['returns'])==9 and np.isfinite(stats['returns']).all()
    assert np.isclose(np.mean(stats['returns']),stats['mean'])
    assert payload['env']==job['target'] and payload['skill_input_mode']=='ego_abs_target'
    write(BATCH/'status'/f"{job['name']}.done.json",{'mean':stats['mean'],'episodes':9})

def coordinate(jobs):
    for name in ('logs','status','eval','csv'):(BATCH/name).mkdir(exist_ok=True)
    running=[];pending=list(jobs);done=[];failed=[]
    free=set(os.sched_getaffinity(0))
    while pending or running:
        for job in list(pending):
            if len(free)<5:break
            allowed={p['proc'].pid for p in running if p['job']['gpu']==job['gpu']}
            processes=subprocess.check_output(['nvidia-smi','-i',str(job['gpu']),'--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True)
            if any(line and int(line.split(',')[0]) not in allowed and 'ollama' not in line.lower() for line in processes.splitlines()):continue
            cpus=sorted(free)[:5];free.difference_update(cpus)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(job['gpu']),JAX_PLATFORMS='cuda,cpu',
                XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='0.078',
                OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',WANDB_MODE='disabled',
                CBL_EVAL_MOD_WORKER='1',SDL_VIDEODRIVER='dummy',MPLCONFIGDIR=str(BATCH/'.matplotlib'),
                PYTHONPATH=f"{ROOT/'src'}:{Path.home()/'JAXAtari-seaquest-robustness/src'}:{ROOT}",
                XLA_FLAGS='--xla_gpu_enable_triton_gemm=false')
            log=(BATCH/'logs'/f"{job['name']}.log").open('w')
            cmd=['taskset','-c',','.join(map(str,cpus)),sys.executable,'-u',str(Path(__file__).resolve()),'worker','--job',job['name']]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            running.append({'job':job,'proc':proc,'log':log,'cpus':cpus,'started':time.time()})
            pending.remove(job);print('START',job['name'],'gpu',job['gpu'],'pid',proc.pid,flush=True);time.sleep(1)
        for p in list(running):
            rc=p['proc'].poll()
            if rc is None:continue
            free.update(p['cpus']);p['log'].close();running.remove(p)
            result={'name':p['job']['name'],'exit_code':rc,'seconds':time.time()-p['started']}
            (done if rc==0 else failed).append(result);print('FINISH',result,flush=True)
        write(BATCH/'status/coordinator.json',{'pending':[j['name'] for j in pending],
            'running':[{'name':p['job']['name'],'pid':p['proc'].pid,'gpu':p['job']['gpu']} for p in running],
            'completed':done,'failures':failed})
        if pending or running:time.sleep(10)
    print('TRANSFER_FINISHED',flush=True)
    return bool(failed)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['coordinate','worker']);parser.add_argument('--job');args=parser.parse_args()
    jobs=json.loads((BATCH/'jobs.json').read_text())
    if args.mode=='worker':worker(next(j for j in jobs if j['name']==args.job))
    else:sys.exit(coordinate(jobs))
