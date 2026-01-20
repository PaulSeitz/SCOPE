import jax.numpy as jnp

# Renumbered Core Concepts (removed LABEL_PLAYER_TOOL)
LABEL_COLLECT = 0
LABEL_AVOID = 1
LABEL_DESTROY = 2
LABEL_DEFAULT = 3  # Monitor / Surface (was 4, now 3)

EVENT_TRIGGERED = 1
EVENT_NOT_TRIGGERED = 0

# --- Capabilities Indices ---
CAP_CAN_FIRE = 0
CAP_FACING_X = 1  # NEW: The x-component of the shooting vector
CAP_FACING_Y = 2  # NEW: The y-component (0 for Seaquest, -1 for Phoenix)
N_CAPABILITIES = 4  # Fixed size for transfer compatibility

