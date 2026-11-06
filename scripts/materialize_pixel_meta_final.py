#!/usr/bin/env python3
"""Prepare final raw-slot tandem configs without touching running suite configs.

Includes optional eval-only aliases for reuse of the accepted Seaquest seed-44
pilot. Scheduler must verify saved-config equivalence before using these aliases.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import yaml


def build(source_root,output_root):
    path=source_root/'scripts/materialize_thesis_final.py'
    spec=importlib.util.spec_from_file_location('pixel_final_recipe',path)
    recipe=importlib.util.module_from_spec(spec);sys.modules[spec.name]=recipe
    spec.loader.exec_module(recipe)
    runs=recipe.build_runs()
    selected=[r for r in runs if r.arm in ('tandem_pixel','tandem_pixel_skills')]
    out=output_root/'configs/experiment/paper_eval/pixel_meta_final'
    out.mkdir(parents=True,exist_ok=True)
    docs={}
    for run in selected:
        stem=run.stem.replace('tandem_pixel','tandem_pixel_slots')
        text=yaml.safe_dump(run.doc,sort_keys=False)
        text=text.replace('tandem_pixel','tandem_pixel_slots')
        text=text.replace('outputs/models/paper_eval/thesis_final/',
                          'outputs/models/paper_eval/pixel_meta_final/')
        doc=yaml.safe_load(text)
        doc.update(META_ARCHITECTURE='pixel_slots',META_INPUT_MODE='pixel_embed',
                   META_TARGET_LOCKED_INPUT=False)
        doc.setdefault('alg',{}).update(META_ARCHITECTURE='pixel_slots',
                META_INPUT_MODE='pixel_embed',META_TARGET_LOCKED_INPUT=False)
        if run.phase=='train':
            doc['alg'].update(META_USE_NOOP_TARGET=True,META_USE_FIRE_TARGET=False)
        tags=[t for t in doc.get('WANDB_TAGS',[]) if not t.startswith('suite:')]
        doc['WANDB_TAGS']=['suite:pixel_meta_final',*tags,'meta_arch:pixel_slots',
                           'label_input:none','annotation_input:none']
        docs[stem]=doc
    # Preserve pilot provenance, rather than copying or renaming its artifacts.
    source_stem='tf_seaquest_tandem_pixel_slots_seed44'
    pilot='pm_seaquest_slots_seed44'
    oldpath=f'./outputs/models/paper_eval/pixel_meta_final/{source_stem}'
    pilotpath=f'./outputs/models/paper_eval/meta_selector_pilots/{pilot}'
    for stem,original in list(docs.items()):
        if original.get('PAPER_EVAL_ARCHITECTURE_OF')!=source_stem:
            continue
        doc=copy.deepcopy(original)
        if original['PAPER_EVAL_KIND']=='tandem':
            alias='ps_seaquest_slots_seed44_full_eval'
        else:
            alias=f'ps_zs_slots_seaquest_to_{original["ENV_NAME"]}_seed44'
        doc['PAPER_EVAL_ARCHITECTURE_OF']=pilot
        doc['PAPER_EVAL_REQUIRES_RQ1']=[pilot]
        for key in ('PAPER_EVAL_SKILLS_CKPT','PAPER_EVAL_META_CKPT'):
            if key in doc:doc[key]=doc[key].replace(oldpath,pilotpath)
        doc['SAVE_PATH']=f'./outputs/models/paper_eval/pixel_meta_final/{alias}'
        doc['WANDB_RUN_NAME']=alias
        doc['WANDB_TAGS']=[*doc['WANDB_TAGS'],'provenance:architecture_selection_pilot']
        docs[alias]=doc
    for stem,doc in docs.items():
        text='# Final raw-image slot selector; scheduling is manual.\n'+yaml.safe_dump(doc,sort_keys=False)
        target=out/f'{stem}.yaml'
        if target.exists() and target.read_text()!=text:
            raise RuntimeError(f'Refusing to overwrite changed final config: {target}')
        target.write_text(text)
    manifest={'architecture':'pixel_slots','seeds':list(recipe.SEEDS),
              'environments':list(recipe.HOME_ENVS),'stems':sorted(docs),
              'source_recipe_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'pilot_reuse':'Seaquest seed44 aliases are conditional on saved-config equivalence; do not count aliases and fresh seed44 as separate seeds.',
              'launch_policy':'manual; do not append to running queue scripts'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Prepared {len(docs)} isolated configs; no jobs launched.')
    return docs


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    parser.add_argument('--source-root',type=Path,default=root)
    parser.add_argument('--output-root',type=Path,default=root)
    args=parser.parse_args();build(args.source_root,args.output_root)
