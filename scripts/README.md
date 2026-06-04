# Scripts

## `queue_train_after.sh`

Queue `python main.py experiment=...` to start after a training PID exits or the GPU is idle.

```bash
# Help
./scripts/queue_train_after.sh --help

# After a specific run (get PID from pgrep)
pgrep -af "python main.py experiment=seaquest_skills_pretrain"
./scripts/queue_train_after.sh --wait-pid 22944 --experiment seaquest_skills_pretrain_vel

# When GPU 0 is free
./scripts/queue_train_after.sh --experiment freeway_fewshot

## Seaquest pretrain A/B configs

| Experiment | Size (hitboxes) | Velocity | Ego dim (stack=4) |
|------------|-----------------|----------|-------------------|
| `seaquest_skills_pretrain` | no | no | 12 |
| `seaquest_skills_pretrain_vel` | no | yes | 20 |
| `seaquest_skills_pretrain_vel_size` | yes | yes | 28 |

# Block terminal instead of background
./scripts/queue_train_after.sh --wait-gpu 0 --experiment seaquest_skills_pretrain_vel --foreground

# Extra Hydra overrides
./scripts/queue_train_after.sh --experiment seaquest_skills_pretrain_vel -- SEED=44 WANDB_MODE=online
```

Logs go to `logs/queue_<experiment>_<timestamp>.log`. Use `tail -f` on that path.

Run from a shell with the correct pyenv/venv active, or pass `--python "$(pyenv which python)"`.
