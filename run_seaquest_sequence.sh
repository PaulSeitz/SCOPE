#!/usr/bin/env bash

set -euo pipefail

echo "Running seaquest_skills experiment..."
python main.py experiment=seaquest_skills

echo "Running seaquest_meta_only experiment..."
python main.py experiment=seaquest_meta_only

echo "Done."
