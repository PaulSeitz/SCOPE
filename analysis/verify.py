#!/usr/bin/env python3
"""Independently verify generated numbers and the recorded SVG-to-PDF outputs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def same(actual, expected):
    assert math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-6), (actual, expected)

def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'assets')
    parser.add_argument('--tables-output', type=Path, default=HERE/'tables')
    args = parser.parse_args()
    out, tables = args.output, args.tables_output
    manifest = json.loads((out/'manifest.json').read_text())
    illustrations=json.loads((out/'modification_manifest.json').read_text())
    for category in ['source_files_sha256','frame_files_sha256']:
        for path,expected in illustrations[category].items():assert digest(path)==expected, f'Illustration input changed: {path}'
    assert illustrations['sample_frames']==[320,324,328,332,336]
    for path, expected in manifest['source_files_sha256'].items():
        assert digest(path) == expected, f'Source changed: {path}; regenerate assets.'
    for name, expected in manifest['source_scripts_sha256'].items():
        assert digest(HERE/name) == expected, f'Generator changed: {name}; regenerate assets.'
    for asset in manifest['assets']:
        svg, pdf = out/asset['svg'], out/asset['pdf']
        assert digest(svg) == asset['svg_sha256'] == asset['pdf_converted_from_svg_sha256']
        assert digest(pdf) == asset['pdf_sha256']
        root = ET.parse(svg).getroot()
        tags = [x.tag.rsplit('}', 1)[-1] for x in root.iter()]
        assert 'text' in tags, f'No editable text: {svg}'
        # The compact method illustrations (render_compact_examples.py, 2026-11-10) embed saved game frames by design.
        assert 'image' not in tags or asset['name'] in ('fig_environments', 'fig_worked_example'), f'Embedded raster: {svg}'
        page = PdfReader(pdf).pages
        assert len(page) == 1, pdf
        # Inkscape page dimensions must match the SVG physical canvas.
        def points(value):
            if value.endswith('pt'): return float(value[:-2])
            if value.endswith('in'): return float(value[:-2])*72
            if value.endswith('mm'): return float(value[:-2])*72/25.4
            raise AssertionError(value)
        assert abs(float(page[0].mediabox.width)-points(root.attrib['width'])) < .02
        assert abs(float(page[0].mediabox.height)-points(root.attrib['height'])) < .02
    groups = defaultdict(list)
    cache = {}
    checkpoint_rows = rows(out/'checkpoint_means.csv')
    for row in checkpoint_rows:
        path = row['path']
        assert path in manifest['source_files_sha256'] and 'nopov' not in Path(path).name
        if path not in cache: cache[path] = json.loads(Path(path).read_text())
        condition = row['coordinates'].split('|')[-1]
        raw = cache[path]['results'][condition]
        assert int(row['n_episodes']) == len(raw['returns']) == 9
        same(row['mean'], raw['mean'])
        same(row['mean'], statistics.mean(raw['returns']))
        groups[row['kind'], row['coordinates']].append((int(row['seed']), float(raw['mean'])))
    summaries = {}
    for row in rows(out/'summary.csv'):
        key = row['kind'], row['coordinates']
        group = sorted(groups[key])
        seeds, values = zip(*group)
        assert len(set(seeds)) == len(seeds) == int(row['n'])
        assert row['seeds'] == ','.join(map(str, seeds))
        same(row['mean'], statistics.mean(values))
        if len(values)>1: same(row['sd'], statistics.stdev(values))
        else: assert not row['sd']
        summaries[key] = row
    assert len(summaries) == len(groups)
    assert len(checkpoint_rows) == manifest['validated_condition_means']
    controls = {}
    for row in rows(out/'control_means.csv'):
        if not row['mean']: continue
        raw = json.loads(Path(row['path']).read_text())['results'][row['mod']]
        assert len(raw['returns']) == int(row['n_episodes']) == 27
        same(row['mean'], raw['mean'])
        same(row['mean'], statistics.mean(raw['returns']))
        # Transfer tables compare against base-condition controls only.
        if row['mod'] == 'base': controls[row['env'], row['kind']] = float(row['mean'])
    def check_summary(row, key, prefix=''):
        source = summaries.get(key)
        if not source:
            assert not row[prefix+'mean'] and int(row[prefix+'n']) == 0
        else:
            same(row[prefix+'mean'], float(source['mean']))
            assert int(row[prefix+'n']) == int(source['n'])
            if source['sd']: same(row[prefix+'sd'], float(source['sd']))
            else: assert not row[prefix+'sd']
    for row in rows(tables/'all_transfer_results.csv'):
        check_summary(row, ('transfer', '|'.join([row['arm'], row['source'], row['target'], 'base'])))
    arms = {'ego':'skills_ego','screen':'skills_screen','pixel':'skills_pixel_outline','abs_target':'skills_ego_abstarget'}
    for row in rows(tables/'skill_transfer.csv'):
        check_summary(row, ('transfer', '|'.join([arms[row['method']], row['source'], row['target'], 'base'])))
    for row in rows(tables/'native_targets.csv'):
        for prefix,key in [('native','ego'),('pqn_pixel','pqn_pixel'),('pqn_oc','pqn_oc')]:
            check_summary(row,('home',row['target']+'|'+key+'|base'),prefix+'_')
    for row in rows(tables/'reference_transfer.csv'):
        target=row['target']
        for kind in ['random_actions','heuristic_random','fixed_ego_rules']:
            value=controls.get((target,kind))
            if value is None: assert not row[kind]
            else: same(row[kind],value)
        for source in ['seaquest','namethisgame']:
            if source==target:
                assert row[source+'_status']=='source_equals_target'
                assert not row[source+'_mean'] and int(row[source+'_n'])==0
            else:
                check_summary(row,('transfer','|'.join(['skills_ego',source,target,'base'])),source+'_')
        check_summary(row,('home',target+'|ego|base'),'native_')
    for row in rows(tables/'tandem_transfer.csv'):
        check_summary(row,('transfer','|'.join([row['arm'],row['source'],row['target'],'base'])))
    for row in rows(tables/'pqn_transfer.csv'):
        assert row['source']=='seaquest'
        check_summary(row,('transfer','|'.join([row['arm'],row['source'],row['target'],'base'])))
    rescue_groups=defaultdict(list)
    for row in rows(tables/'rescue_checkpoint_means.csv'):
        assert row['path'] in manifest['source_files_sha256']
        condition=json.loads(Path(row['path']).read_text())['results'][row['condition']]
        counts=[float(ep['env_rescues']) for ep in condition['episode_metrics']]
        assert len(counts)==int(row['n_episodes'])==9
        assert all(math.isfinite(v) and v>=0 for v in counts)
        same(row['mean_rescues'],statistics.mean(counts))
        same(row['mean_rescues'],condition['mean_env_rescues'])
        rescue_groups[row['method'],row['condition']].append((int(row['seed']),statistics.mean(counts)))
    for row in rows(tables/'rescue_summary.csv'):
        group=rescue_groups[row['method'],row['condition']]
        assert len({seed for seed,value in group})==int(row['n'])==5
        same(row['mean'],statistics.mean(value for seed,value in group))
        same(row['sd'],statistics.stdev(value for seed,value in group))
    counter_groups=defaultdict(list)
    for row in rows(tables/'seaquest_counter_checkpoints.csv'):
        assert row['metric'] in ('env_rescues','env_steps')
        assert row['path'] in manifest['source_files_sha256']
        raw=json.loads(Path(row['path']).read_text())['results'][row['condition']]
        counts=[float(ep[row['metric']]) for ep in raw['episode_metrics']]
        assert len(counts)==int(row['n_episodes'])==9
        assert all(math.isfinite(v) and v>=0 for v in counts)
        same(row['mean'],statistics.mean(counts))
        same(row['mean'],raw['mean_'+row['metric']])
        counter_groups[row['method'],row['condition'],row['metric']].append((int(row['seed']),statistics.mean(counts)))
    for row in rows(tables/'seaquest_counter_summary.csv'):
        group=counter_groups[row['method'],row['condition'],row['metric']]
        assert row['metric'] in ('env_rescues','env_steps')
        assert len({seed for seed,value in group})==len(group)==int(row['n'])==5
        same(row['mean'],statistics.mean(value for seed,value in group))
        if len(group)>1:same(row['sd'],statistics.stdev(value for seed,value in group))
    pong_groups=defaultdict(list)
    for row in rows(tables/'pong_episodes.csv'):
        assert row['path'] in manifest['source_files_sha256']
        raw=json.loads(Path(row['path']).read_text())['results']['base']
        index=int(row['episode'])-1
        same(row['return'],raw['returns'][index])
        same(row['env_steps'],raw['episode_metrics'][index]['env_steps'])
        assert int(row['near_cap'])==int(float(row['env_steps'])>=26900)
        assert int(row['return_21'])==int(float(row['return'])==21)
        pong_groups[row['method'],row['seed']].append(row)
    pong_methods=defaultdict(list)
    for row in rows(tables/'pong_checkpoints.csv'):
        group=pong_groups[row['method'],row['seed']]
        assert len(group)==int(row['n_episodes'])==9
        same(row['mean_return'],statistics.mean(float(e['return']) for e in group))
        same(row['mean_env_steps'],statistics.mean(float(e['env_steps']) for e in group))
        for key in ['near_cap','return_21']:assert int(row[key])==sum(int(e[key]) for e in group)
        pong_methods[row['method']].append(row)
    for row in rows(tables/'pong_summary.csv'):
        group=pong_methods[row['method']]
        assert len(group)==5 and len({e['seed'] for e in group})==5
        same(row['mean_return'],statistics.mean(float(e['mean_return']) for e in group))
        same(row['sd_return'],statistics.stdev(float(e['mean_return']) for e in group))
        same(row['mean_env_steps'],statistics.mean(float(e['mean_env_steps']) for e in group))
        for key in ['near_cap','return_21','n_episodes']:assert int(row[key])==sum(int(e[key]) for e in group)
    assert len(PdfReader(out/'evaluation_assets_review.pdf').pages) == len(manifest['assets'])
    print(f'Verified {len(manifest["assets"])} SVG/PDF pairs, {len(checkpoint_rows)} raw condition means, {len(summaries)} summaries, controls and table exports.')

if __name__ == '__main__':
    main()
