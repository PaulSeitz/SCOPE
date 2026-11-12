"""Config and source-geometry contracts for the three interface probes."""
from pathlib import Path
import yaml
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs/experiment/paper_eval/thesis_final"

@pytest.mark.parametrize("variant,early,geometry", [
    ("a_early", True, "centre_ray"),
    ("b_closest", False, "closest_points"),
    ("c_combined", True, "closest_points"),
])
def test_probe_configs_preserve_recipe_and_source_geometry(variant, early, geometry):
    manifest = yaml.safe_load((OUT / "_seaquest_interface_probes_manifest.yaml").read_text())
    info = manifest["variants"][variant]
    assert len(info["runs_in_order"]) == 10
    train = info["train"]
    for i,stem in enumerate(info["runs_in_order"]):
        raw = yaml.safe_load((OUT / f"{stem}.yaml").read_text())
        cfg = {**raw, **raw["alg"]}
        assert stem in cfg["SAVE_PATH"]
        assert stem in cfg["WANDB_RUN_NAME"]
        assert cfg["EGO_AABB_GEOM"] == geometry
        assert cfg["COLLECT_EARLY_WHEN_NO_THREATS"] is (early if i < 2 else False)
        assert cfg["AVOID_KILLED_EVENT_MODE"] == "none"
        assert cfg["AVOID_SPARSE_ONLY"] is False
        assert cfg["AVOID_ESCAPE_BONUS"] == 0.01
        assert cfg["COLLECT_DEATH_PRECEDENCE"] == "death_first"
        assert cfg["COLLECT_SPARSE_ONLY"] is True
        assert cfg["SKILL_DEATH_PENALIZE_COLLECT"] is True
        assert cfg["OFF_POLICY"] is True
        assert cfg["FRAME_SKIP"] == cfg["FRAME_STACK_SIZE"] == 4
        assert cfg["EGO_PSI_ANGLE_BINS"] == 8
        if i == 0:
            assert cfg["TOTAL_TIMESTEPS"] == 50_000_000
            assert train in cfg["SAVE_PATH"]
        else:
            assert raw["PAPER_EVAL_ARCHITECTURE_OF"] == train
            assert raw["PAPER_EVAL_REQUIRES_RQ1"] == [train]
            assert train in raw["PAPER_EVAL_SKILLS_CKPT"]
            assert raw["EVAL_SEED"] == 10044 and raw["EVAL_NUM_SEEDS"] == 3
            if i >= 2:
                assert raw["PAPER_EVAL_MOD_KEYS"] == ["base"]

def test_checkpoint_overlay_preserves_closest_point_geometry(tmp_path):
    from concept_based_learning.paper_eval.eval_harness import _overlay_ego_flags_from_skill_config
    model = tmp_path / "skills.safetensors"
    (tmp_path / "skills_config.yaml").write_text(
        yaml.safe_dump({"EGO_AABB_GEOM": "closest_points"})
    )
    cfg = _overlay_ego_flags_from_skill_config(
        {"EGO_AABB_GEOM": "centre_ray"}, model
    )
    assert cfg["EGO_AABB_GEOM"] == "closest_points"


def test_probe_output_paths_do_not_collide():
    manifest = yaml.safe_load((OUT / "_seaquest_interface_probes_manifest.yaml").read_text())
    paths = []
    for info in manifest["variants"].values():
        for stem in info["runs_in_order"]:
            raw = yaml.safe_load((OUT / f"{stem}.yaml").read_text())
            cfg = {**raw, **raw["alg"]}
            paths.append(cfg["SAVE_PATH"])
    assert len(paths) == 30
    assert len(set(paths)) == 30
