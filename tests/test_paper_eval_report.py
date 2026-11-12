"""Smoke tests for paper_eval report ingest + retention + PDF build."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from concept_based_learning.paper_eval.report.build import build_paper_eval_report
from concept_based_learning.paper_eval.report.load import load_mod_eval_summary
from concept_based_learning.paper_eval.report.metrics import (
    base_returns_by_agent,
    mean_retention_by_category,
    retention_matrix,
)


def _write_summary(path: Path) -> None:
    fieldnames = [
        "agent_kind",
        "env",
        "skill_input_mode",
        "config_stem",
        "paper_eval_rq",
        "paper_eval_kind",
        "object_centric",
        "mod_key",
        "mod_category",
        "mean",
        "std",
        "drop_frac_from_base",
        "mean_env_score_final",
        "std_env_score_final",
    ]
    rows = [
        # skills ego — clipped mean, unclipped score (report must prefer score)
        dict(
            agent_kind="skills",
            env="seaquest",
            skill_input_mode="ego",
            config_stem="seaquest_skills_ego_eval",
            paper_eval_rq=1,
            paper_eval_kind="skills",
            object_centric="",
            mod_key="base",
            mod_category="control",
            mean=100,
            std=5,
            drop_frac_from_base=0,
            mean_env_score_final=5000,
            std_env_score_final=50,
        ),
        dict(
            agent_kind="skills",
            env="seaquest",
            skill_input_mode="ego",
            config_stem="seaquest_skills_ego_eval",
            paper_eval_rq=1,
            paper_eval_kind="skills",
            object_centric="",
            mod_key="no_enemies",
            mod_category="semantic",
            mean=80,
            std=4,
            drop_frac_from_base=-0.2,
            mean_env_score_final=4000,
            std_env_score_final=40,
        ),
        dict(
            agent_kind="skills",
            env="seaquest",
            skill_input_mode="ego",
            config_stem="seaquest_skills_ego_eval",
            paper_eval_rq=1,
            paper_eval_kind="skills",
            object_centric="",
            mod_key="faster_enemies",
            mod_category="dynamics",
            mean=50,
            std=3,
            drop_frac_from_base=-0.5,
            mean_env_score_final=2500,
            std_env_score_final=30,
        ),
        # pqn pixel — mean already equals score
        dict(
            agent_kind="flat_pqn",
            env="seaquest",
            skill_input_mode="",
            config_stem="seaquest_pqn_eval",
            paper_eval_rq=1,
            paper_eval_kind="flat_pqn",
            object_centric="0",
            mod_key="base",
            mod_category="control",
            mean=200,
            std=10,
            drop_frac_from_base=0,
            mean_env_score_final=200,
            std_env_score_final=10,
        ),
        dict(
            agent_kind="flat_pqn",
            env="seaquest",
            skill_input_mode="",
            config_stem="seaquest_pqn_eval",
            paper_eval_rq=1,
            paper_eval_kind="flat_pqn",
            object_centric="0",
            mod_key="no_enemies",
            mod_category="semantic",
            mean=40,
            std=8,
            drop_frac_from_base=-0.8,
            mean_env_score_final=40,
            std_env_score_final=8,
        ),
        dict(
            agent_kind="flat_pqn",
            env="seaquest",
            skill_input_mode="",
            config_stem="seaquest_pqn_eval",
            paper_eval_rq=1,
            paper_eval_kind="flat_pqn",
            object_centric="0",
            mod_key="faster_enemies",
            mod_category="dynamics",
            mean=20,
            std=5,
            drop_frac_from_base=-0.9,
            mean_env_score_final=20,
            std_env_score_final=5,
        ),
        # heuristic — clipped mean + score
        dict(
            agent_kind="heuristic_random",
            env="seaquest",
            skill_input_mode="ego",
            config_stem="seaquest_heuristic_random",
            paper_eval_rq=1,
            paper_eval_kind="heuristic_random",
            object_centric="",
            mod_key="base",
            mod_category="control",
            mean=10,
            std=1,
            drop_frac_from_base=0,
            mean_env_score_final=150,
            std_env_score_final=5,
        ),
        dict(
            agent_kind="heuristic_random",
            env="seaquest",
            skill_input_mode="ego",
            config_stem="seaquest_heuristic_random",
            paper_eval_rq=1,
            paper_eval_kind="heuristic_random",
            object_centric="",
            mod_key="no_enemies",
            mod_category="semantic",
            mean=9,
            std=1,
            drop_frac_from_base=-0.1,
            mean_env_score_final=135,
            std_env_score_final=4,
        ),
        dict(
            agent_kind="heuristic_random",
            env="seaquest",
            skill_input_mode="ego",
            config_stem="seaquest_heuristic_random",
            paper_eval_rq=1,
            paper_eval_kind="heuristic_random",
            object_centric="",
            mod_key="faster_enemies",
            mod_category="dynamics",
            mean=8,
            std=1,
            drop_frac_from_base=-0.2,
            mean_env_score_final=120,
            std_env_score_final=3,
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_and_retention(tmp_path: Path) -> None:
    csv_path = tmp_path / "all_mod_eval_summary.csv"
    _write_summary(csv_path)
    rows = load_mod_eval_summary(csv_path)
    assert len(rows) == 9
    bases = dict(base_returns_by_agent(rows))
    assert "skills/ego" in bases
    # Reported return is the unclipped episode mean, not the engine-score column.
    assert bases["skills/ego"][0] == pytest.approx(100)
    assert bases["pqn/pixel"][0] == pytest.approx(200)
    cats = mean_retention_by_category(rows)
    assert cats["skills/ego"]["semantic"] == pytest.approx(0.8)  # 4000/5000
    assert cats["pqn/pixel"]["semantic"] == pytest.approx(0.2)
    mods, agents, matrix = retention_matrix(rows)
    assert "no_enemies" in mods
    assert "skills/ego" in agents


def test_prefers_score_over_clipped_mean(tmp_path: Path) -> None:
    csv_path = tmp_path / "all_mod_eval_summary.csv"
    _write_summary(csv_path)
    rows = load_mod_eval_summary(csv_path, rq=1)
    ego_base = next(
        r for r in rows if r["agent_label"] == "skills/ego" and r["mod_key"] == "base"
    )
    assert ego_base["mean"] == pytest.approx(100)
    assert ego_base["score_source"] == "episode_return"
    assert ego_base["mean_reward_sum"] == pytest.approx(100)


def test_keeps_mean_when_env_score_is_zero(tmp_path: Path) -> None:
    """Pixel wrappers can leave env_score_final=0 while mean is the true return."""
    csv_path = tmp_path / "pixel_score0.csv"
    fieldnames = [
        "agent_kind",
        "env",
        "skill_input_mode",
        "config_stem",
        "paper_eval_rq",
        "paper_eval_kind",
        "object_centric",
        "mod_key",
        "mod_category",
        "mean",
        "std",
        "drop_frac_from_base",
        "mean_env_score_final",
        "std_env_score_final",
        "max_frames",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(
            dict(
                agent_kind="skills",
                env="seaquest",
                skill_input_mode="pixel_embed",
                config_stem="seaquest_skills_pixel_eval",
                paper_eval_rq=1,
                paper_eval_kind="skills",
                object_centric="",
                mod_key="base",
                mod_category="control",
                mean=4473,
                std=100,
                drop_frac_from_base=0,
                mean_env_score_final=0,
                std_env_score_final=0,
                max_frames=27000,
            )
        )
    rows = load_mod_eval_summary(csv_path)
    assert rows[0]["mean"] == pytest.approx(4473)
    assert rows[0]["score_source"] == "episode_return"


def test_dedupes_to_longest_horizon(tmp_path: Path) -> None:
    csv_path = tmp_path / "mixed_horizon.csv"
    fieldnames = [
        "agent_kind",
        "env",
        "skill_input_mode",
        "config_stem",
        "paper_eval_rq",
        "paper_eval_kind",
        "object_centric",
        "mod_key",
        "mod_category",
        "mean",
        "std",
        "drop_frac_from_base",
        "mean_env_score_final",
        "std_env_score_final",
        "max_frames",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for frames, score in ((800, 3000), (27000, 22000)):
            w.writerow(
                dict(
                    agent_kind="flat_pqn",
                    env="seaquest",
                    skill_input_mode="",
                    config_stem="seaquest_pqn_eval",
                    paper_eval_rq=1,
                    paper_eval_kind="flat_pqn",
                    object_centric="0",
                    mod_key="base",
                    mod_category="control",
                    mean=score,
                    std=100,
                    drop_frac_from_base=0,
                    mean_env_score_final=score,
                    std_env_score_final=100,
                    max_frames=frames,
                )
            )
    rows = load_mod_eval_summary(csv_path)
    assert len(rows) == 1
    assert rows[0]["mean"] == pytest.approx(22000)
    assert rows[0]["max_frames"] == pytest.approx(27000)


def test_build_pdf_rq1(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    csv_path = tmp_path / "all_mod_eval_summary.csv"
    _write_summary(csv_path)
    out = tmp_path / "report.pdf"
    build_paper_eval_report(csv_path, output_pdf=out, fig_dir=tmp_path / "figs", rq=1)
    assert out.is_file() and out.stat().st_size > 1000
    assert (tmp_path / "figs" / "rq1_base_returns.png").is_file()


def test_train_peak_vs_eval(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from concept_based_learning.paper_eval.report.training_metrics import (
        discover_training_peaks,
        train_eval_pairs,
    )

    # Synthetic train tree: .../rq1/seaquest_skills_ego/seaquest/training_metrics.csv
    train_csv = (
        tmp_path / "rq1" / "seaquest_skills_ego" / "seaquest" / "training_metrics.csv"
    )
    train_csv.parent.mkdir(parents=True)
    train_csv.write_text(
        "update_steps,charts/avg_game_return\n"
        "1,100\n"
        "2,9000\n"
        "3,7000\n",
        encoding="utf-8",
    )
    pqn_csv = tmp_path / "rq1" / "seaquest_pqn" / "seaquest" / "training_metrics.csv"
    pqn_csv.parent.mkdir(parents=True)
    pqn_csv.write_text(
        "update_steps,returned_episode_returns\n1,500\n2,2500\n",
        encoding="utf-8",
    )

    peaks = discover_training_peaks(tmp_path)
    assert peaks["skills/ego"]["peak"] == pytest.approx(9000)
    assert peaks["pqn/pixel"]["peak"] == pytest.approx(2500)

    csv_path = tmp_path / "all_mod_eval_summary.csv"
    _write_summary(csv_path)
    rows = load_mod_eval_summary(csv_path)
    pairs = {p["agent_label"]: p for p in train_eval_pairs(rows, peaks, rq=1)}
    assert pairs["skills/ego"]["train_peak"] == pytest.approx(9000)
    assert pairs["skills/ego"]["eval_mean"] == pytest.approx(100)
    assert pairs["skills/ego"]["gap"] == pytest.approx(8900)

    out = tmp_path / "report_train.pdf"
    build_paper_eval_report(
        csv_path,
        output_pdf=out,
        fig_dir=tmp_path / "figs2",
        rq=1,
        train_root=tmp_path,
    )
    assert (tmp_path / "figs2" / "rq1_train_vs_eval.png").is_file()
