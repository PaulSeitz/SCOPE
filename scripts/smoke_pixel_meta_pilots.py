#!/usr/bin/env python3
"""Two tiny real-environment tandem updates; writes only to an explicit scratch path.

This tests wiring/checkpoint save, not competence. Full pilot budgets remain in YAML.
"""
from pathlib import Path
import argparse
import importlib.util
import sys
import yaml

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    parser.add_argument('config',type=Path)
    parser.add_argument('--project-root',type=Path,default=root)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    args.output_root.mkdir(parents=True,exist_ok=True)
    doc=yaml.safe_load(args.config.read_text());cfg={**doc,**doc.get('alg',{})};cfg.pop('alg',None)
    cfg.update(NUM_ENVS=2,NUM_STEPS=4,NUM_MINIBATCHES=1,NUM_EPOCHS=1,
               TOTAL_TIMESTEPS=16,TOTAL_TIMESTEPS_DECAY=16,
               META_HIDDEN_SIZE=8,META_EMBED_DIM=8,META_FLAT_NUM_LAYERS=1,
               META_PIXEL_CNN_CHANNELS=[2],META_PIXEL_CNN_KERNEL_SIZE=[3,3],
               META_PIXEL_RESIZE_SHAPE=[24,24],
               HIDDEN_SIZE=8,NUM_LAYERS=1,WANDB_MODE='disabled',RTPT_ENABLED=False,
               RECORD_VIDEO=False,VIDEO_AT_END=False,SAVE_TRAINING_METRICS_CSV=False,
               SAVE_PATH=str(args.output_root.resolve()),NUM_SEEDS=1)
    # Imports the regular entrypoint while respecting the caller's PYTHONPATH
    # (which can point at a staged source tree during pre-install validation).
    spec=importlib.util.spec_from_file_location('pilot_training_main',args.project_root/'main.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    module.single_run(cfg)
    meta=list(args.output_root.rglob('*_meta.safetensors'))
    skills=list(args.output_root.rglob('*_skills.safetensors'))
    if len(meta)!=1 or len(skills)!=1:
        raise RuntimeError('Smoke run did not produce both tandem checkpoints.')
    print('PASS: two real-environment updates and both checkpoint saves.')
