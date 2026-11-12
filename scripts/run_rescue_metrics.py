#!/usr/bin/env python3
"""Collect Seaquest rescue metrics separately from all frozen score artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def worker(job):
    import numpy as np
    from nexus.mod_eval import run_nexus_mod_eval
    for rel,expected in job['checkpoint_files_sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
    # Names and seed construction are the unchanged NEXUS frozen evaluator's.
    payload=run_nexus_mod_eval(nexus_root=ROOT/'third_party/symbolic_options',
        jaxatari_root=Path.home()/'JAXAtari-seaquest-robustness',alg=job['alg'],
        ckpt_dir=ROOT/job['checkpoint'],seed=10000+job['seed'],num_episodes=3,num_seeds=3,
        max_frames=27000,mod_keys=[job['mod']],output_json=ROOT/'eval_jobs'/job['name']/'result.json',
        record_episode_metrics=True)
    result=payload['results'][job['mod']]
    assert payload['status']=='complete' and len(result['returns'])==len(result['episode_metrics'])==9
    rescues=[ep['env_rescues'] for ep in result['episode_metrics']]
    assert all(np.isfinite(x) and x>=0 and float(x).is_integer() for x in rescues)
    assert np.isclose(np.mean(rescues),result['mean_env_rescues'])
    assert all(int(ep['env_steps'])==length for ep,length in zip(result['episode_metrics'],result['episode_lengths']))
    assert np.isfinite(result['returns']).all()
    payload['rescue_collection']={'source_frozen_eval':job['frozen_eval'],'source_frozen_eval_sha256':job['frozen_eval_sha256'],
        'checkpoint_files_sha256':job['checkpoint_files_sha256'],'training_seed':job['seed'],
        'counter':'sum of positive successful_rescues increments, finalized as in SCOPE',
        'record_episode_metrics':True}
    write(ROOT/'eval_jobs'/job['name']/'result.json',payload)
    write(ROOT/'status'/f"{job['name']}.done.json",{'mean_return':result['mean'],'mean_env_rescues':result['mean_env_rescues'],
        'returns_match_frozen':bool(np.array_equal(result['returns'],job['frozen_returns'])),'episodes':9})
    print('RESCUE_JOB_OK',job['name'],'rescues',rescues,'returns_match_frozen',np.array_equal(result['returns'],job['frozen_returns']),flush=True)

def coordinate(jobs):
    for name in ('logs','status','eval_jobs','eval'):(ROOT/name).mkdir(exist_ok=True)
    pending=list(jobs);running=[];done=[];failed=[];free=set(os.sched_getaffinity(0))
    while pending or running:
        for job in list(pending):
            if len(free)<5:break
            selected=None
            for gpu in (4,5,6,7,0,1,2,3):
                if sum(p['gpu']==gpu for p in running)>=(2 if gpu>=4 else 1):continue
                allowed={p['proc'].pid for p in running if p['gpu']==gpu}
                lines=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).splitlines()
                if any(line and int(line.split(',')[0]) not in allowed and 'ollama' not in line.lower() for line in lines):continue
                selected=gpu;break
            if selected is None:break
            cpus=sorted(free)[:5];free.difference_update(cpus)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(selected),JAX_PLATFORMS='cuda,cpu',
                XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='0.15',
                OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',WANDB_MODE='disabled',
                CBL_EVAL_MOD_WORKER='1',SDL_VIDEODRIVER='dummy',MPLCONFIGDIR=str(ROOT/'.matplotlib'),
                PYTHONPATH=f"{ROOT/'src'}:{ROOT/'scripts'}:{ROOT/'third_party/symbolic_options/src'}:{Path.home()/'JAXAtari-seaquest-robustness/src'}",
                XLA_FLAGS='--xla_gpu_enable_triton_gemm=false')
            log=(ROOT/'logs'/f"{job['name']}.log").open('w')
            cmd=['taskset','-c',','.join(map(str,cpus)),sys.executable,'-u',str(Path(__file__).resolve()),'worker','--job',job['name']]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            running.append({'job':job,'gpu':selected,'proc':proc,'cpus':cpus,'log':log,'started':time.time()});pending.remove(job)
            print('START',job['name'],'gpu',selected,'pid',proc.pid,flush=True);time.sleep(1)
        for p in list(running):
            rc=p['proc'].poll()
            if rc is None:continue
            p['log'].close();free.update(p['cpus']);running.remove(p)
            item={'name':p['job']['name'],'exit_code':rc,'seconds':time.time()-p['started']}
            (done if rc==0 else failed).append(item);print('FINISH',item,flush=True)
        write(ROOT/'status/coordinator.json',{'pending':[j['name'] for j in pending],
            'running':[{'name':p['job']['name'],'gpu':p['gpu'],'pid':p['proc'].pid} for p in running],
            'completed':done,'failures':failed})
        if pending or running:time.sleep(10)
    print('RESCUE_COLLECTION_FINISHED',flush=True)
    return bool(failed)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['coordinate','worker']);parser.add_argument('--job');args=parser.parse_args()
    jobs=json.loads((ROOT/'jobs.json').read_text())
    if args.mode=='worker':worker(next(j for j in jobs if j['name']==args.job))
    else:sys.exit(coordinate(jobs))
