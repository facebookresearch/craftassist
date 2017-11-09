import time
import agent

DEFAULT_PORT = 25565


def agent_init(port=DEFAULT_PORT):
    """ initialize the agent on localhost at specified port"""
    return agent.Agent("localhost", port, default_agent_name())


def default_agent_name():
    """Use a unique name based on timestamp"""
    return "bot.{}".format(str(time.time())[3:13])
