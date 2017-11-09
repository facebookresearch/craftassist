import atexit

from agent import Agent
from agent_connection import default_agent_name
from cuberite_process import CuberiteProcess

# Launch server
p = CuberiteProcess("flat_world", seed=0, game_mode="survival", plugins=["starting_inventory"])
atexit.register(lambda: p.destroy())  # close Cuberite server when this script exits

# Launch agent
agent = Agent("localhost", 25565, default_agent_name())

# Craft pickaxe (block id 270)
r = agent.craft(270)
assert r > 0, r

# Hold pickaxe in hand
assert agent.set_held_item(270)

print("Done! Agent should be holding a pickaxe")

# ctrl+C to exit
while True:
    pass
