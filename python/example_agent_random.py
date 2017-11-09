"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import random
import atexit

from agent import Agent
from cuberite_process import CuberiteProcess

random.seed(0)

p = CuberiteProcess("flat_world", seed=0, game_mode="creative")
atexit.register(lambda: p.destroy())  # close Cuberite server when this script exits

agent = Agent("localhost", 25565, "example_bot")

# Random walk forever
while True:
    random.choice([agent.step_pos_x, agent.step_pos_z, agent.step_neg_x, agent.step_neg_z])()
