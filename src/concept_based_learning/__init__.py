"""
Public package entrypoint for Concept-Based Learning.

Typical usage:

    import concept_based_learning as cbl
    config = cbl.load_config_from_file("configs/default.yaml")
    from concept_based_learning.environments import labeling_wrapper
"""

from concept_based_learning.config import (
    Config,
    ModelConfig,
    TrainingConfig,
    RewardConfig,
    EnvironmentConfig,
    ConceptConstants,
    WandBConfig,
    get_default_config,
    load_config_from_file,
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
]

