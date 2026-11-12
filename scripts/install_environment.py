#!/usr/bin/env python3
"""Install SCOPE and its pinned upstream JAXtari base in an active environment."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "https://github.com/k4ntz/JAXAtari.git"
REVISION = "ae1d7a0a2146f09bd6bdbb1bebf21ce4f3e9dd94"
PATCH = ROOT / "third_party/jaxtari/evaluation-compat.patch"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda12", action="store_true", help="Install the CUDA 12 JAX extras")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        parser.error("Run from a virtual environment (python -m venv .venv).")
    with tempfile.TemporaryDirectory(prefix="scope-jaxtari-") as temp:
        def git(*args):
            subprocess.run(["git", *args], cwd=temp, check=True)

        git("init", "--quiet")
        git("fetch", "--quiet", "--depth=1", UPSTREAM, REVISION)
        git("checkout", "--quiet", "--detach", "FETCH_HEAD")
        git("apply", "--check", str(PATCH))
        git("apply", str(PATCH))
        cmd = [sys.executable, "-m", "pip", "install", "--constraint",
               str(ROOT / "requirements-eval.txt"), "--force-reinstall", temp,
               f"{ROOT}[analysis]"]
        if args.cuda12:
            cmd.append("jax[cuda12]")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
