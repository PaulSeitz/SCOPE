# Re-export shim — primary code lives in label.py.
from concept_based_learning.environments.seaquest.label import (  # noqa: F401
    extract_all_skills_rewards,
    seaquest_action_mapper,
    SeaquestSemanticWrapper,
)
