import jax.numpy as jnp

LABEL_PLAYER_TOOL = 0
LABEL_COLLECT = 1
LABEL_AVOID = 2
LABEL_DESTROY = 3
LABEL_MONITOR_STATISTIC = 4 # this is very specific to Seaquest, it will label the oxygen statistic. We should look at a better way to do such high level monitoring, for example using BlendRL. 
# Our approach concerns itself mainly with enabling low level decision making, so high level concepts are in part left to the roadside. Maybe use a general high level concept space for all environments, tests tbd
LABEL_HIGH_LEVEL_CONCEPT = 5 # could be anything game specific really

EVENT_TRIGGERED = 1
EVENT_NOT_TRIGGERED = 0

# --- Capabilities Indices ---
CAP_CAN_FIRE = 0
CAP_FACING_X = 1  # NEW: The x-component of the shooting vector
CAP_FACING_Y = 2  # NEW: The y-component (0 for Seaquest, -1 for Phoenix)
N_CAPABILITIES = 4  # Fixed size for transfer compatibility

