#!/usr/bin/env bash
# Steps 1+2 for Seaquest + Phoenix final_10m skills re-run at SKILL_LR=5e-4.
# Run ON cluster1 from the repo root (or: ssh cluster1 'bash -s' < this_script).
set -euo pipefail

ROOT="${ROOT:-$HOME/ConceptBasedLearning}"
CFG="$ROOT/configs/experiment/paper_eval/final_10m"
OUT="$ROOT/outputs/models/paper_eval/final_10m"
EVAL="$ROOT/outputs/models/paper_eval/eval"

echo "=== host=$(hostname) root=$ROOT ==="

# --- Step 1: stash collapsed ckpts ---
mkdir -p "$OUT" "$EVAL/_lr1e3_collapsed"
cd "$OUT"
for d in final_10m_skills_ego final_10m_skills_ego_phoenix; do
  if [ -d "$d" ]; then
    if [ -d "${d}_lr1e3_collapsed" ]; then
      echo "ALREADY STASHED: ${d}_lr1e3_collapsed — removing live $d"
      rm -rf "$d"
    else
      mv "$d" "${d}_lr1e3_collapsed"
      echo "MOVED: $d -> ${d}_lr1e3_collapsed"
    fi
  else
    echo "MISSING CKPT DIR (ok): $d"
  fi
done

# Stash old eval artifacts so --resume won't treat them as done
cd "$EVAL"
for stem in final_10m_skills_ego_eval final_10m_skills_ego_phoenix_eval; do
  shopt -s nullglob
  for f in ${stem}_mods*; do
    mv "$f" "_lr1e3_collapsed/$f"
    echo "MOVED EVAL: $f -> _lr1e3_collapsed/"
  done
  shopt -u nullglob
done

# --- Step 2: patch configs (seaquest + phoenix train+eval) ---
python3 - <<'PY'
from pathlib import Path
import re
import os

cfg = Path(os.environ.get("ROOT", Path.home() / "ConceptBasedLearning")) / \
    "configs/experiment/paper_eval/final_10m"
files = [
    "final_10m_skills_ego.yaml",
    "final_10m_skills_ego_eval.yaml",
    "final_10m_skills_ego_phoenix.yaml",
    "final_10m_skills_ego_phoenix_eval.yaml",
]
for name in files:
    p = cfg / name
    text = p.read_text()
    orig = text
    text = text.replace("locked lr1e3 + ε B", "SKILL_LR=5e-4 + ε B (smoke recipe re-run)")
    text = text.replace("lock:skills_lr1e3", "lock:skills_lr5e4")
    text = text.replace("  SKILL_LR: 0.001\n", "  SKILL_LR: 0.0005\n")
    if text == orig:
        # already patched?
        print(f"NO DIFF: {name}")
    else:
        p.write_text(text)
        print(f"PATCHED: {name}")
    lr = re.search(r"SKILL_LR:\s*([0-9.eE+-]+)", text)
    lock = "lock:skills_lr5e4" in text
    print(f"  check {name}: SKILL_LR={lr.group(1) if lr else 'n/a'} lock_lr5e4={lock}")
PY

echo "=== verify ==="
ls -d "$OUT"/final_10m_skills_ego* 2>/dev/null || true
ls "$EVAL"/_lr1e3_collapsed/final_10m_skills_ego* 2>/dev/null | head -20 || true
grep -n 'SKILL_LR\|lock:skills' \
  "$CFG"/final_10m_skills_ego.yaml \
  "$CFG"/final_10m_skills_ego_phoenix.yaml \
  "$CFG"/final_10m_skills_ego_eval.yaml \
  "$CFG"/final_10m_skills_ego_phoenix_eval.yaml
echo DONE
