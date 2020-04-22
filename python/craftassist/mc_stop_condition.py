"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from base_agent.stop_condition import StopCondition


class AgentAdjacentStopCondition(StopCondition):
    def __init__(self, agent, bid):
        super().__init__(agent)
        self.bid = bid
        self.name = "adjacent_block"

    def check(self):
        B = self.agent.get_local_blocks(1)
        return (B[:, :, :, 0] == self.bid).any()
