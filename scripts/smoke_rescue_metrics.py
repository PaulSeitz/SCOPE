"""One short real-checkpoint check for NEXUS rescue logging."""
import json
from pathlib import Path
from nexus.mod_eval import run_nexus_mod_eval
ROOT=Path(__file__).resolve().parents[1]
job=json.loads((ROOT/'jobs.json').read_text())[0]
payload=run_nexus_mod_eval(nexus_root=ROOT/'third_party/symbolic_options',jaxatari_root=Path.home()/'JAXAtari-seaquest-robustness',
    alg=job['alg'],ckpt_dir=ROOT/job['checkpoint'],seed=10044,num_episodes=1,num_seeds=1,max_frames=64,
    mod_keys=['base'],output_json=ROOT/'smoke/result.json',record_episode_metrics=True)
stats=payload['results']['base']
assert payload['status']=='complete'
assert len(stats['returns'])==len(stats['episode_metrics'])==1
assert stats['episode_metrics'][0]['env_steps']==stats['episode_lengths'][0]
assert stats['episode_metrics'][0]['env_rescues']>=0
print('RESCUE_SMOKE_PASSED',stats['episode_metrics'],flush=True)
