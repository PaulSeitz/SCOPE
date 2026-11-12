from main import _wandb_env_step


def test_wandb_env_step_matches_training_metrics_convention():
    config = {"NUM_STEPS": 32, "NUM_ENVS": 2048}
    assert _wandb_env_step(config, timesteps=1_000_000) == 1_000_000
    assert (
        _wandb_env_step(config, timesteps=500_000, step_offset_updates=10)
        == 10 * 32 * 2048 + 500_000
    )
