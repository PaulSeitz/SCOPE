"""
Public package entrypoint for Concept-Based Learning.

Typical usage:

    import concept_based_learning as cbl
    config = cbl.load_config_from_file("configs/default.yaml")
    cbl.init_wandb(config)   # one-line wandb setup
    from concept_based_learning.environments import labeling_wrapper
"""

from .config import (
    Config,
    ModelConfig,
    TrainingConfig,
    RewardConfig,
    EnvironmentConfig,
    ConceptConstants,
    WandBConfig,
    get_default_config,
    load_config_from_file,
    init_wandb,
)

__all__ = [
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "RewardConfig",
    "EnvironmentConfig",
    "ConceptConstants",
    "WandBConfig",
    "get_default_config",
    "load_config_from_file",
    "init_wandb",
]

