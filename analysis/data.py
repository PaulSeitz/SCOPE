"""Load and validate the approved run artifacts without modifying them."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import math
import re
import statistics

ENVIRONMENTS = ['seaquest', 'namethisgame', 'spaceinvaders', 'beamrider', 'asteroids', 'pong', 'breakout', 'freeway', 'phoenix']
HOMES = ENVIRONMENTS[:2]
TARGET_ORDER = ['namethisgame', 'seaquest', *ENVIRONMENTS[2:]]
NAMES = dict(zip(ENVIRONMENTS, ['Seaquest', 'Name This Game', 'Space Invaders', 'Beam Rider', 'Asteroids', 'Pong', 'Breakout', 'Freeway', 'Phoenix']))
ARMS = {'skills_ego':'ego', 'skills_screen':'screen', 'skills_ocnative':'oc', 'skills_pixel_outline':'pixel', 'pqn_pixstack':'pqn_pixel', 'pqn_ocnative':'pqn_oc', 'pqn_oc_h128l3':'pqn_oc_large', 'skills_ego_nocross':'nocross', 'tandem_ocnative':'tandem_oc', 'tandem_pixel_slots':'tandem_pixel', 'skills_ego_scenerel':'scene', 'skills_ego_scenerows':'rows', 'skills_ego_abspos':'abspos', 'skills_ego_abstarget':'abs_target', 'skills_ego_ocnative':'native_ref', 'skills_ocnative_targetfront':'native_fixed', 'pqn_ego_scenerel':'pqn_scene'}
RESCUE_RERUN = None  # set by generate.py --rescue-root
RESCUE_RERUN_ARMS = {'nexus_symbolic_pixel':'symbolic_pixel', 'hrl_pixel_optim50m':'hrl_pixel', 'hrl_oc_optim50m':'hrl_oc'}
CONTROL_KINDS = ['random_actions', 'heuristic_random', 'fixed_ego_rules']
MODS = {'seaquest':['base','random_spawns','shift_lanes','gravity','faster_enemies','no_enemies','change_background_color'], 'namethisgame':['base','shift_tentacle_spawn_x','slower_player','faster_shark','no_shark','no_shark_short_tentacles','change_background_color']}

@dataclass(frozen=True)
class Observation:
    seed: int
    mean: float
    returns: tuple[float, ...]
    path: str

@dataclass(frozen=True)
class Summary:
    mean: float | None
    sd: float | None
    n: int
    seeds: tuple[int, ...]
    values: tuple[float, ...]

def summarize(rows):
    rows = sorted(rows, key=lambda x:x.seed)
    if not rows:
        return Summary(None, None, 0, (), ())
    seeds = tuple(x.seed for x in rows)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f'Duplicate training seeds: {seeds}')
    values = tuple(x.mean for x in rows)
    return Summary(statistics.mean(values), statistics.stdev(values) if len(values)>1 else None, len(values), seeds, values)

def checked_condition(stats, expected, path, mod):
    """Use the stored episode mean; reject inconsistent/nonfinite artifacts."""
    if 'mean' not in stats or 'returns' not in stats:
        raise ValueError(f'{path}: {mod}: missing mean/returns')
    mean = float(stats['mean'])
    values = tuple(float(v) for v in stats['returns'])
    if len(values) != expected:
        raise ValueError(f'{path}: {mod}: {len(values)} returns, expected {expected}')
    if not math.isfinite(mean) or not all(math.isfinite(v) for v in values):
        raise ValueError(f'{path}: {mod}: nonfinite return')
    if not math.isclose(mean, statistics.mean(values), rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f'{path}: {mod}: stored mean differs from episode mean')
    return mean, values

class Runs:
    def __init__(self, results_root, controls_root, overrides):
        root = Path(results_root).resolve()
        self.root = root/'eval' if (root/'eval').is_dir() else root
        self.controls_root = Path(controls_root).resolve()
        self.overrides = Path(overrides).resolve()
        self.home = defaultdict(list)
        self.transfer = defaultdict(list)
        self.sources = {}
        self.controls = {}
        self.excluded = []
        self.unrecognized = []
        self.missing = set()
        self.audited_means = 0
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        for path in sorted(self.root.glob('*_mods.json')):
            self._load(path)
        self.rescue_rerun = self._load_rescue_rerun()

    def _load_rescue_rerun(self):
        """Seaquest rescue counters for NEXUS symbolic pixel and HRL, measured in a separate rerun (2026-11-12).
        Returns stay on the frozen evaluations; only the counters are taken from here (provenance kept explicit)."""
        out = defaultdict(list)
        if RESCUE_RERUN is None or not Path(RESCUE_RERUN).is_dir():
            raise FileNotFoundError(f'Rescue-metric rerun not found: {RESCUE_RERUN}')
        for path in sorted(Path(RESCUE_RERUN).glob('cbl_seaquest_*_mods.json')):
            m = re.fullmatch(r'cbl_seaquest_(nexus_symbolic_pixel|hrl_pixel_optim50m|hrl_oc_optim50m)_seed(\d+)_mods.json', path.name)
            if not m:
                continue
            key = RESCUE_RERUN_ARMS[m.group(1)]
            d = self._read(path)
            for condition in ('base', 'no_enemies'):
                eps = d['results'][condition]['episode_metrics']
                if len(eps) != 9 or any('env_rescues' not in ep for ep in eps):
                    raise ValueError(f'Invalid rescue rerun: {path}/{condition}')
                out[key, condition].append(Observation(int(m.group(2)), statistics.mean(float(ep['env_rescues']) for ep in eps), tuple(float(ep['env_rescues']) for ep in eps), str(path.resolve())))
        for k, v in out.items():
            if len(v) != 5:
                raise ValueError(f'Rescue rerun requires five checkpoints: {k}')
        return out

    def _read(self, path):
        raw = path.read_bytes()
        d = json.loads(raw)
        if d.get('status') == 'failed' or d.get('failed_mods'):
            raise ValueError(f'Failed/incomplete artifact: {path}')
        self.sources[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return d

    def _load(self, path):
        name = path.name
        if 'nopov' in name:
            self.excluded.append({'path':str(path), 'reason':'author-excluded legacy target-aligned no-POV'})
            return
        if name.startswith('tf_floor_'):
            return  # Loaded separately using explicit precedence.
        transfer = re.fullmatch(r'tf_zs_(.*?)_(seaquest|namethisgame)_to_(.*?)_seed(\d+)_mods.json', name)
        local = re.fullmatch(r'tf_([a-z]+)_(.*?)_seed(\d+)_eval_mods.json', name)
        nexus = re.fullmatch(r'cbl_(seaquest|namethisgame)_(nexus_(neural|nesy|symbolic)_(oc|pixel)|hrl_(oc|pixel)_optim50m|hrl_baseline_optim50m)_seed(\d+)_mods.json', name)
        if transfer:
            arm, src, dst, seed = transfer.groups()
            if arm not in ['skills_ego','skills_ego_abstarget','skills_screen','skills_pixel_outline','tandem_ocnative_skills','tandem_pixel_slots_skills','pqn_pixstack']:
                self.unrecognized.append(str(path)); return
            destination = self.transfer
            prefix = (arm, src, dst)
        elif local and local.group(2) in ARMS and local.group(1) in ENVIRONMENTS:
            env, arm, seed = local.groups(); destination = self.home; prefix = (env, ARMS[arm])
        elif nexus:
            env, family, policy, modality, hrl_modality, seed = nexus.groups()
            key = f'{policy}_{modality}' if policy else 'hrl_'+(hrl_modality or 'oc')
            destination = self.home; prefix = (env, key)
        else:
            self.unrecognized.append(str(path)); return
        d = self._read(path)
        expected = int(d.get('num_episodes',3))*int(d.get('num_seeds',3))
        if expected != 9:
            raise ValueError(f'{path}: expected learned evaluation to contain nine episodes, got {expected}')
        for mod, stats in d['results'].items():
            mean, returns = checked_condition(stats, 9, path, mod)
            destination[(*prefix, mod)].append(Observation(int(seed),mean,returns,str(path.resolve())))
            self.audited_means += 1

    def base(self, env, method, mod='base'):
        result = summarize(self.home[env,method,mod])
        if not result.n: self.missing.add(f'home:{env}:{method}:{mod}')
        return result

    def transferred(self, arm, src, dst, mod='base'):
        result = summarize(self.transfer[arm,src,dst,mod])
        if not result.n and src != dst:self.missing.add(f'transfer:{arm}:{src}:{dst}:{mod}')
        return result

    def control(self, env, kind, mod='base'):
        key = (env,kind,mod)
        if key in self.controls:return self.controls[key]
        stem=f'tf_floor_{kind}_{env}_mods.json'
        # Never silently use the superseded October-15 Seaquest controller floors.
        candidates=[self.overrides/stem,self.root/stem]
        if not (env=='seaquest' and kind in ['heuristic_random','fixed_ego_rules']):
            candidates.append(self.controls_root/stem)
        path=next((p for p in candidates if p.is_file()),None)
        if path is None:
            self.missing.add(f'control:{env}:{kind}:{mod}'); self.controls[key]=None; return None
        d=self._read(path)
        if mod not in d['results']:
            self.missing.add(f'control:{env}:{kind}:{mod}'); self.controls[key]=None; return None
        mean,values=checked_condition(d['results'][mod],27,path,mod)
        self.controls[key]={'mean':mean,'n_episodes':len(values),'path':str(path.resolve())}
        return self.controls[key]

    def normalized_rows(self):
        for collection, groups in [('home',self.home),('transfer',self.transfer)]:
            for key, observations in sorted(groups.items()):
                for obs in sorted(observations,key=lambda x:x.seed):
                    yield {'kind':collection,'coordinates':'|'.join(key),'seed':obs.seed,'mean':obs.mean,'n_episodes':len(obs.returns),'path':obs.path}

    def summaries(self):
        for collection, groups in [('home',self.home),('transfer',self.transfer)]:
            for key, observations in sorted(groups.items()):
                if not observations:continue
                s=summarize(observations)
                yield {'kind':collection,'coordinates':'|'.join(key),'mean':s.mean,'sd':s.sd,'n':s.n,'seeds':','.join(map(str,s.seeds))}
