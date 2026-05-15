# Re-export shim — this network is deferred; canonical copy lives in _legacy/.
from concept_based_learning._legacy.affordance_pointer_network import (  # noqa: F401
    MetaTransition,
    MetaQNetwork,
    flatten_raw_observation,
)
