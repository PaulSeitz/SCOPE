"""Tests for mod-eval text summaries + wandb_run.json persistence."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from concept_based_learning.paper_eval.mod_eval_report import (
    format_mod_eval_summary_text,
    publish_mod_eval_report,
    resolve_evaluated_ckpt_dir,
    sync_mod_eval_summary_to_train_wandb,
    write_mod_eval_summary_files,
)
from concept_based_learning.paper_eval.wandb_run_meta import (
    load_wandb_run_meta,
    persist_wandb_run_meta,
    wandb_run_meta_path,
)


def _payload():
    return {
        "env_name": "seaquest",
        "results": {
            "base": {"mean": 100.0, "std": 1.0, "category": "control"},
            "gravity": {
                "mean": 80.0,
                "std": 2.0,
                "category": "dynamics",
                "drop_from_base": -20.0,
            },
        },
    }


def test_persist_and_load_wandb_run_meta(tmp_path: Path):
    run = MagicMock()
    run.id = "abc123"
    run.name = "sq_train"
    run.project = "concept_based_learning"
    run.entity = "me"
    run.path = "me/concept_based_learning/abc123"
    run.url = "https://wandb.ai/me/concept_based_learning/runs/abc123"
    path = persist_wandb_run_meta(tmp_path, run)
    assert path is not None
    assert path == wandb_run_meta_path(tmp_path)
    meta = load_wandb_run_meta(tmp_path)
    assert meta["id"] == "abc123"
    assert meta["project"] == "concept_based_learning"


def test_persist_wandb_run_meta_noop_without_run(tmp_path: Path):
    assert persist_wandb_run_meta(tmp_path, None) is None
    assert load_wandb_run_meta(tmp_path) is None


def test_format_and_write_summary(tmp_path: Path):
    text = format_mod_eval_summary_text(
        _payload(),
        extra_meta={"config_stem": "demo_eval", "env_name": "seaquest"},
    )
    assert "base" in text
    assert "gravity" in text
    assert "AVG_OVER_MODS" in text
    paths = write_mod_eval_summary_files(
        _payload(),
        tmp_path,
        config_stem="demo_eval",
        extra_meta={"config_stem": "demo_eval"},
        home_eval=True,
    )
    assert len(paths) == 2
    assert (tmp_path / "mod_eval_demo_eval_summary.txt").is_file()
    assert (tmp_path / "mod_eval_summary.txt").is_file()
    assert "100.00" in paths[0].read_text(encoding="utf-8")


def test_resolve_ckpt_dir_from_skills_path(tmp_path: Path):
    ckpt = tmp_path / "seaquest" / "hierarchical_pqn_seaquest_seed44.safetensors"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")
    got = resolve_evaluated_ckpt_dir(skills_path=ckpt)
    assert got == ckpt.parent.resolve()


def test_publish_without_wandb_meta_still_writes(tmp_path: Path):
    ckpt = tmp_path / "seaquest" / "model.safetensors"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")
    paths = publish_mod_eval_report(
        _payload(),
        skills_path=ckpt,
        env_name="seaquest",
        config_stem="home_eval",
        extra_meta={"config_stem": "home_eval"},
        wandb_mode="disabled",
    )
    assert paths
    assert (ckpt.parent / "mod_eval_summary.txt").is_file()


def test_sync_to_wandb_resumes_train_run_id(tmp_path: Path):
    class _Run:
        id = "trainrun99"
        name = "sq_paper"
        project = "concept_based_learning"
        entity = None
        path = None
        url = None

    persist_wandb_run_meta(tmp_path, _Run())
    assert load_wandb_run_meta(tmp_path)["id"] == "trainrun99"

    summary = tmp_path / "mod_eval_summary.txt"
    summary.write_text("hello\n", encoding="utf-8")
    text = summary.read_text(encoding="utf-8")

    import sys
    import types

    fake_run = MagicMock()
    fake_run.summary = {}
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = MagicMock(return_value=fake_run)
    fake_wandb.summary = {}
    fake_art = MagicMock()
    fake_wandb.Artifact = MagicMock(return_value=fake_art)
    fake_wandb.finish = MagicMock()
    with patch.dict(sys.modules, {"wandb": fake_wandb}):
        ok = sync_mod_eval_summary_to_train_wandb(
            ckpt_dir=tmp_path,
            summary_paths=[summary],
            payload=_payload(),
            config_stem="home_eval",
            text=text,
            wandb_mode="offline",
        )
    assert ok is True
    fake_wandb.init.assert_called_once()
    kwargs = fake_wandb.init.call_args.kwargs
    assert kwargs["id"] == "trainrun99"
    assert kwargs["resume"] == "allow"
    assert fake_run.summary["mod_eval/home_eval/base/mean"] == 100.0
    fake_wandb.finish.assert_called()


def test_sync_skips_cleanly_when_disabled(tmp_path: Path):
    class _Run:
        id = "x"
        name = "n"
        project = "p"
        entity = None
        path = None
        url = None

    persist_wandb_run_meta(tmp_path, _Run())
    summary = tmp_path / "mod_eval_summary.txt"
    summary.write_text("t\n", encoding="utf-8")
    ok = sync_mod_eval_summary_to_train_wandb(
        ckpt_dir=tmp_path,
        summary_paths=[summary],
        payload=_payload(),
        config_stem="x",
        text="t",
        wandb_mode="disabled",
    )
    assert ok is False
