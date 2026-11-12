#!/usr/bin/env python3
"""Read the detached absolute-target campaign status without initializing JAX."""
import argparse
import json
import shlex
import subprocess

REMOTE = r'''
import csv,json
from pathlib import Path
root=Path.home()/'ConceptBasedLearning/outputs/abs_target_campaign_20261111'
state=json.loads((root/'status'/f'{HOST}.json').read_text())
rows=[]
for job in state['running']:
    row={k:job[k] for k in ('game','seed','gpu','mode','pid')}
    if 'mod' in job:row['mod']=job['mod']
    metric=root/f"outputs/models/paper_eval/abs_target_ablation/tf_{job['game']}_skills_ego_abstarget_seed{job['seed']}/{job['game']}/training_metrics.csv"
    if metric.exists():
        try:
            with metric.open() as f:last=list(csv.DictReader(f))[-1]
            row['env_steps']=last.get('env/step');row['updates']=last.get('update_steps')
        except (IndexError,KeyError):pass
    rows.append(row)
print(json.dumps({'host':HOST,'updated':state['updated'],'running':rows,
                  'pending_training':len(state['pending_training']),
                  'pending_evaluation':len(state['pending_evaluation']),
                  'completed':len(state['completed']),'failures':state['failures'],
                  'aggregated_results':len(list((root/'eval').glob('*_mods.json')))},indent=2))
'''
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--host',choices=['cluster1','cluster2','both'],default='both')
args=parser.parse_args()
for host in (['cluster1','cluster2'] if args.host=='both' else [args.host]):
    code=f'HOST={host!r}\n'+REMOTE
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',host,
                           'python3 -c '+shlex.quote(code)],capture_output=True,text=True)
    if result.returncode:print(host,result.stderr.strip() or result.stdout.strip())
    else:print(result.stdout.strip())
