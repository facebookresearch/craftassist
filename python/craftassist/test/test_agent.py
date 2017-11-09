import unittest

from craftassist_agent import CraftAssistAgent


class BaseAgentTest(unittest.TestCase):
    def test_init_agent(self):
        CraftAssistAgent()


if __name__ == "__main__":
    unittest.main()
