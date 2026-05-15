"""Egocentric actions → global ALE indices for Kinematic Bootcamp.

Bootcamp supports full 8-directional movement with fire, using the same Atari
integer layout as Seaquest (NOOP=0, FIRE=1, UP=2, RIGHT=3, ..., DOWNLEFTFIRE=17).
The target-aligned mapper computes θ_target from player→target geometry and maps
ego FORWARD to the nearest world direction toward the target.
"""

from concept_based_learning.observations.action_space import make_ego_action_mapper

bootcamp_action_mapper = make_ego_action_mapper(
    noop_action=0,
    fire_action=1,
    move_table=     [3, 8, 5, 9, 4, 7, 2, 6],
    move_fire_table=[11, 16, 13, 17, 12, 15, 10, 14],
)
