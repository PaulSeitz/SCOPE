#!/usr/bin/env python3
"""Configs for the frozen post-reward-fix wave.

Does not rewrite the running ego or channel-stacked pixel PQN queues.
New stems are used wherever an old checkpoint would make --resume skip a
corrected retrain that must not reuse the old observation or the old reward.
Screen, no-POV, pixel-outline, and pixel-slot keep their stems; those
checkpoints are archived separately so --resume retrains them.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PAPER = _ROOT / "configs" / "experiment" / "paper_eval"
_OUT = _PAPER / "rewardfix_wave"
_ENVS = ("seaquest", "namethisgame")
_SEEDS = (44, 45, 46, 47, 48)
_TARGETS = (
    "seaquest",
    "namethisgame",
    "spaceinvaders",
    "beamrider",
    "asteroids",
    "pong",
    "breakout",
    "freeway",
    "phoenix",
)


def _targets(src: str) -> tuple[str, ...]:
    return tuple(e for e in _TARGETS if e != src)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _reseed(text: str, old: int, new: int) -> str:
    text = text.replace(f"100{old}", f"100{new}")
    text = text.replace(f"seed{old}", f"seed{new}")
    # ``SEED: 46`` is not covered by the stem replace above.
    text = text.replace(f"SEED: {old}\n", f"SEED: {new}\n")
    return text


def _clone_family(needle: str, *, src_seed: int = 46) -> int:
    """Copy every seed-46 yaml whose name contains ``needle`` to seeds 47 and 48."""
    n = 0
    for src in sorted(_PAPER.rglob(f"*seed{src_seed}*.yaml")):
        if needle not in src.name:
            continue
        if _OUT in src.parents:
            continue
        body = _read(src)
        for seed in (47, 48):
            dest = _OUT / src.name.replace(f"seed{src_seed}", f"seed{seed}")
            _write(dest, _reseed(body, src_seed, seed))
            n += 1
    if n == 0:
        raise SystemExit(f"no seed-{src_seed} yamls matched {needle!r}")
    return n


def _insert_native_stack(text: str) -> str:
    if "OC_NATIVE_STACK:" in text:
        return text
    lines = []
    inserted = False
    for line in text.splitlines(keepends=True):
        lines.append(line)
        if not inserted and line.startswith("ENV_NAME:"):
            lines.append("OC_NATIVE_STACK: true\n")
            inserted = True
    if not inserted:
        raise SystemExit("ENV_NAME missing while adding OC_NATIVE_STACK")
    return "".join(lines)


def _ocnative_tandem() -> int:
    n = 0
    produced: dict[str, str] = {}
    for src in sorted(_PAPER.rglob("*tandem_oc*seed4*.yaml")):
        if "tandem_ocnative" in src.name or _OUT in src.parents:
            continue
        if "ocraw" in src.name or "oc_raw" in str(src):
            continue
        # Only the role-pooled thesis_final selector, not pixel or other arms.
        if "thesis_final" not in str(src):
            continue
        text = _read(src)
        text = text.replace("tandem_oc", "tandem_ocnative")
        text = text.replace("META_ARCHITECTURE: per_object_mlp", "META_ARCHITECTURE: flat_mlp")
        text = text.replace("META_INPUT_MODE: oc_simple_rel_role", "META_INPUT_MODE: oc_native")
        text = _insert_native_stack(text)
        produced[src.name.replace("tandem_oc", "tandem_ocnative")] = text
        n += 1
    # Seeds 47 and 48 from the rewritten seed-46 files.
    extra = {}
    for name, text in produced.items():
        if "seed46" not in name:
            continue
        for seed in (47, 48):
            extra[_reseed(name, 46, seed)] = _reseed(text, 46, seed)
    produced.update(extra)
    for name, text in produced.items():
        _write(_OUT / name, text)
    return len(produced)


def _from_ego(src_name: str, *, kind: str) -> str:
    src = _PAPER / "thesis_final" / src_name
    text = _read(src)
    if kind == "ocnative":
        text = text.replace("skills_ego", "skills_ocnative")
        text = text.replace("SKILL_INPUT_MODE: ego", "SKILL_INPUT_MODE: oc_native")
        text = text.replace("skill_input:ego", "skill_input:oc_native")
        text = _insert_native_stack(text)
        dest = src_name.replace("skills_ego", "skills_ocnative")
    elif kind == "nocross":
        text = text.replace("skills_ego_seed", "skills_ego_nocross_seed")
        text = text.replace("OFF_POLICY: true", "OFF_POLICY: false")
        dest = src_name.replace("skills_ego_seed", "skills_ego_nocross_seed")
    else:
        raise SystemExit(kind)
    _write(_OUT / dest, text)
    return dest


def _ocnative_skills_and_nocross() -> tuple[int, int]:
    skills = 0
    for env in _ENVS:
        for seed in _SEEDS:
            for suffix in ("", "_eval"):
                _from_ego(f"tf_{env}_skills_ego_seed{seed}{suffix}.yaml", kind="ocnative")
                skills += 1
    nocross = 0
    for seed in _SEEDS:
        for suffix in ("", "_eval"):
            _from_ego(f"tf_seaquest_skills_ego_seed{seed}{suffix}.yaml", kind="nocross")
            nocross += 1
    return skills, nocross


def _ocnative_pqn() -> int:
    n = 0
    templates = {
        "seaquest": (
            _PAPER / "oc_raw_pilot" / "ocraw_seaquest_pqn_seed44.yaml",
            _PAPER / "oc_raw_pilot" / "ocraw_seaquest_pqn_seed44_full_eval.yaml",
        ),
        "namethisgame": (
            _PAPER / "oc_raw_pilot" / "ocraw_namethisgame_pqn_seed44.yaml",
            _PAPER / "oc_raw_pilot" / "ocraw_namethisgame_pqn_seed44_full_eval.yaml",
        ),
    }
    for env, (train_src, eval_src) in templates.items():
        train = _read(train_src)
        ev = _read(eval_src)
        for seed in _SEEDS:
            train_s = _reseed(train, 44, seed)
            ev_s = _reseed(ev, 44, seed)
            old = f"ocraw_{env}_pqn_seed{seed}"
            new = f"tf_{env}_pqn_ocnative_seed{seed}"
            for label, text in (("train", train_s), ("eval", ev_s)):
                text = text.replace(f"oc_raw_pilot/{old}", f"rewardfix_wave/{new}")
                text = text.replace(old, new)
                suffix = "" if label == "train" else "_eval"
                _write(_OUT / f"{new}{suffix}.yaml", text)
                n += 1
    return n


def _chain(stems: list[str]) -> list[str]:
    return stems


def _skill_chain(arm: str, env: str, seed: int, *, transfers: bool) -> list[str]:
    out = [f"tf_{env}_{arm}_seed{seed}", f"tf_{env}_{arm}_seed{seed}_eval"]
    if transfers:
        for dst in _targets(env):
            out.append(f"tf_zs_{arm}_{env}_to_{dst}_seed{seed}")
    return out


def _slots_chain(env: str, seed: int) -> list[str]:
    out = [
        f"tf_{env}_tandem_pixel_slots_seed{seed}",
        f"tf_{env}_tandem_pixel_slots_seed{seed}_eval",
    ]
    for dst in _targets(env):
        out.append(f"tf_zs_tandem_pixel_slots_skills_{env}_to_{dst}_seed{seed}")
    return out


def _tandem_chain(env: str, seed: int) -> list[str]:
    out = [
        f"tf_{env}_tandem_ocnative_seed{seed}",
        f"tf_{env}_tandem_ocnative_seed{seed}_eval",
    ]
    for dst in _targets(env):
        out.append(f"tf_zs_tandem_ocnative_skills_{env}_to_{dst}_seed{seed}")
    return out


def _nexus(alg: str, seed: int) -> list[str]:
    root = "$HOME/ConceptBasedLearning"
    save = f"{root}/outputs/models/nexus/{alg}_seed{seed}"
    out_json = (
        f"{root}/outputs/models/paper_eval/eval/{alg}_seed{seed}_mods.json"
    )
    return [
        (
            f"{root}/.venv/bin/python -u scripts/run_nexus.py train "
            f"--alg {alg} --seed {seed} --num-seeds 1 "
            f"--save-path {save} --wandb-mode online"
        ),
        (
            f"{root}/.venv/bin/python -u scripts/run_nexus.py mod-eval "
            f"--alg {alg} --seed {seed + 10000} --eval-episodes 3 "
            f"--eval-num-seeds 3 --eval-max-frames 27000 "
            f"--ckpt {save} --out-json {out_json}"
        ),
    ]


def _paper(stem: str) -> str:
    return (
        "$PY -u scripts/run_paper_eval.py --only "
        f"{stem} --resume --abort-on-failure --lean"
    )


def _flatten_paper(chains: list[list[str]]) -> list[str]:
    return [_paper(stem) for chain in chains for stem in chain]


def build_queues() -> dict[str, list[str]]:
    q: dict[str, list[str]] = {}

    def add(gpu: str, commands: list[str]) -> None:
        q.setdefault(gpu, []).extend(commands)

    # Missing corrected Name This Game → Seaquest ego transfers.
    for gpu, seed in (("c1g0", 44), ("c1g1", 45), ("c1g3", 46), ("c2g0", 47), ("c2g1", 48)):
        add(gpu, [_paper(f"tf_zs_skills_ego_namethisgame_to_seaquest_seed{seed}")])

    add("c1g0", _flatten_paper([
        *[_skill_chain("skills_pixel_outline", "seaquest", s, transfers=True) for s in (44, 45)],
        *[_slots_chain("seaquest", s) for s in (44, 45, 46)],
        *[_skill_chain("skills_pixel_outline", "namethisgame", s, transfers=True) for s in (47, 48)],
    ]))
    add("c1g0", [c for s in (44, 45, 46) for c in _nexus("cbl_seaquest_nexus_neural_pixel", s)])

    add("c1g1", _flatten_paper([
        *[_skill_chain("skills_pixel_outline", "seaquest", s, transfers=True) for s in (46, 47)],
        *[_slots_chain("seaquest", s) for s in (47, 48)],
        *[_slots_chain("namethisgame", s) for s in (44, 45)],
    ]))
    add("c1g1", [c for s in (47, 48) for c in _nexus("cbl_seaquest_nexus_neural_pixel", s)])
    add("c1g1", [c for s in (44,) for c in _nexus("cbl_seaquest_nexus_nesy_pixel", s)])

    add("c1g3", _flatten_paper([
        _skill_chain("skills_pixel_outline", "seaquest", 48, transfers=True),
        *[_skill_chain("skills_pixel_outline", "namethisgame", s, transfers=True) for s in (44, 45, 46)],
        *[_slots_chain("namethisgame", s) for s in (46, 47, 48)],
    ]))
    add("c1g3", [c for s in (45, 46, 47, 48) for c in _nexus("cbl_seaquest_nexus_nesy_pixel", s)])

    add("c2g0", _flatten_paper([
        *[_skill_chain("skills_screen", env, s, transfers=True) for env in _ENVS for s in _SEEDS],
        *[_skill_chain("skills_nopov", "seaquest", s, transfers=True) for s in _SEEDS],
    ]))
    add("c2g0", [c for s in _SEEDS for c in _nexus("cbl_seaquest_nexus_neural_oc", s)])
    add("c2g0", [c for s in (44, 45) for c in _nexus("cbl_seaquest_hrl_pixel_optim50m", s)])

    add("c2g1", _flatten_paper([
        *[_skill_chain("skills_nopov", "namethisgame", s, transfers=True) for s in _SEEDS],
        *[_tandem_chain(env, s) for env in _ENVS for s in _SEEDS],
        *[_skill_chain("skills_ocnative", env, s, transfers=False) for env in _ENVS for s in _SEEDS],
    ]))
    add("c2g1", [c for s in (46, 47, 48) for c in _nexus("cbl_seaquest_hrl_pixel_optim50m", s)])

    add("c2g2", [c for s in _SEEDS for c in _nexus("cbl_seaquest_nexus_symbolic_pixel", s)])

    add("c2g3", _flatten_paper([
        *[_skill_chain("skills_ego_nocross", "seaquest", s, transfers=False) for s in _SEEDS],
        *[
            [f"tf_{env}_pqn_ocnative_seed{s}", f"tf_{env}_pqn_ocnative_seed{s}_eval"]
            for env in _ENVS for s in _SEEDS
        ],
    ]))
    add("c2g3", [c for s in _SEEDS for c in _nexus("cbl_seaquest_hrl_oc_optim50m", s)])
    add("c2g3", [c for s in _SEEDS for c in _nexus("cbl_seaquest_nexus_nesy_oc", s)])
    add("c2g3", [c for s in _SEEDS for c in _nexus("cbl_seaquest_nexus_symbolic_oc", s)])
    return q


def main() -> None:
    if _OUT.exists():
        for old in _OUT.glob("*.yaml"):
            old.unlink()
    n_screen = _clone_family("skills_screen")
    n_nopov = _clone_family("skills_nopov")
    n_outline = _clone_family("skills_pixel_outline")
    n_slots = _clone_family("tandem_pixel_slots")
    n_tandem = _ocnative_tandem()
    n_skills, n_nocross = _ocnative_skills_and_nocross()
    n_pqn = _ocnative_pqn()
    queues = build_queues()
    print(
        f"wrote screen47/48={n_screen} nopov={n_nopov} outline={n_outline} "
        f"slots={n_slots} ocnative_tandem={n_tandem} ocnative_skills={n_skills} "
        f"nocross={n_nocross} ocnative_pqn={n_pqn}"
    )
    for gpu, cmds in queues.items():
        print(f"{gpu}: {len(cmds)} commands")


if __name__ == "__main__":
    main()
