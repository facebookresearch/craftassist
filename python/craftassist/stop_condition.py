class StopCondition:
    def __init__(self, agent):
        self.agent = agent

    def check(self) -> bool:
        raise NotImplementedError("Implemented by subclass")


class NeverStopCondition(StopCondition):
    def __init__(self, agent):
        super().__init__(agent)

    def check(self):
        return False


class AgentAdjacentStopCondition(StopCondition):
    def __init__(self, agent, bid):
        super().__init__(agent)
        self.bid = bid

    def check(self):
        B = self.agent.get_local_blocks(1)
        return (B[:, :, :, 0] == self.bid).any()
