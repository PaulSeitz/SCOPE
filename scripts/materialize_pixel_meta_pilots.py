#!/usr/bin/env python3
"""Prepare isolated seed-44 raw-image / candidate-marked selector pilots.

Clones current tandem pixel settings; never rewrites the thesis-final suite.
"""
from pathlib import Path
import argparse
import copy
import json
import yaml


def build(source_root, output_root):
    source=source_root/'configs/experiment/paper_eval/thesis_final'
    out=output_root/'configs/experiment/paper_eval/meta_selector_pilots'
    out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for env in ('seaquest','namethisgame'):
        template=f'tf_{env}_tandem_pixel_seed44'
        train=yaml.safe_load((source/f'{template}.yaml').read_text())
        evaluate=yaml.safe_load((source/f'{template}_eval.yaml').read_text())
        for arch in ('pixel_slots','pixel_candidates'):
            name=arch.removeprefix('pixel_')
            stem=f'pm_{env}_{name}_seed44'
            target=f'./outputs/models/paper_eval/meta_selector_pilots/{stem}'
            for phase,original in (('train',train),('eval',evaluate)):
                doc=copy.deepcopy(original)
                job=stem if phase=='train' else stem+'_eval'
                doc.update(META_ARCHITECTURE=arch,META_INPUT_MODE='pixel_embed',
                           META_TARGET_LOCKED_INPUT=False,
                           META_CANDIDATE_CHUNK_SIZE=1,
                           SAVE_PATH=target if phase=='train' else target+'_eval',
                           WANDB_RUN_NAME=job,PAPER_EVAL_MOD_KEYS=['base'])
                doc['WANDB_TAGS']=[f'suite:meta_selector_pilots',f'env:{env}',
                                   f'meta_arch:{arch}','seed:44',f'phase:{phase}',
                                   'stage:feasibility','label_input:none',
                                   'annotation_input:' + ('none' if arch=='pixel_slots' else 'target_marker')]
                doc.setdefault('alg',{}).update(
                    META_ARCHITECTURE=arch,META_INPUT_MODE='pixel_embed',
                    META_TARGET_LOCKED_INPUT=False,META_CANDIDATE_CHUNK_SIZE=1,
                    META_USE_NOOP_TARGET=True,META_USE_FIRE_TARGET=False)
                if phase=='train':
                    doc['PAPER_EVAL_REQUIRES_RQ1']=[]
                else:
                    doc['PAPER_EVAL_REQUIRES_RQ1']=[stem]
                    doc['PAPER_EVAL_ARCHITECTURE_OF']=stem
                    prefix=f'{target}/{env}/hierarchical_pqn_{env}_seed44'
                    doc['PAPER_EVAL_SKILLS_CKPT']=prefix+'_skills.safetensors'
                    doc['PAPER_EVAL_META_CKPT']=prefix+'_meta.safetensors'
                text='# Pixel-only selector feasibility pilot; not final results.\n'+yaml.safe_dump(doc,sort_keys=False)
                path=out/f'{job}.yaml'
                if path.exists() and path.read_text()!=text:
                    raise RuntimeError(f'Refusing to overwrite changed pilot config: {path}')
                path.write_text(text)
                manifest.append(job)
    (out/'manifest.json').write_text(json.dumps({'stems':manifest,
        'source_templates':['tf_seaquest_tandem_pixel_seed44',
                            'tf_namethisgame_tandem_pixel_seed44'],
        'launch_policy':'manual scheduler approval; slots first, candidates conditional'},indent=2)+'\n')
    print(f'Prepared {len(manifest)} configs in {out}; no jobs launched.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    parser.add_argument('--source-root',type=Path,default=root)
    parser.add_argument('--output-root',type=Path,default=root)
    args=parser.parse_args()
    build(args.source_root,args.output_root)
