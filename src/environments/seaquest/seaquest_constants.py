LABEL_PLAYER = 0
LABEL_PLAYER_TOOL = 1
LABEL_COLLECT = 2
LABEL_AVOID = 3
LABEL_DESTROY = 4
LABEL_MONITOR_STATISTIC = 5 # this is very specific to Seaquest, it will label the oxygen statistic. We should look at a better way to do such high level monitoring, for example using BlendRL. 
# Our approach concerns itself mainly with enabling low level decision making, so high level concepts are in part left to the roadside. Maybe use a general high level concept space for all environments, tests tbd
LABEL_HIGH_LEVEL_CONCEPT = 6 # could be anything game specific really

EVENT_TRIGGERED = 1
EVENT_NOT_TRIGGERED = 0