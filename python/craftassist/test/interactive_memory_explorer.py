"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import shapes
from base_craftassist_test_case import BaseCraftassistTestCase
from typing import List
from mc_util import Block
from all_test_commands import *  # noqa


def add_two_cubes(test):
    triples = {"has_name": "cube", "has_shape": "cube"}
    test.cube_right: List[Block] = list(
        test.add_object(
            xyzbms=shapes.cube(bid=(42, 0)), origin=(9, 63, 4), relations=triples
        ).blocks.items()
    )
    test.cube_left: List[Block] = list(
        test.add_object(xyzbms=shapes.cube(), origin=(9, 63, 10), relations=triples).blocks.items()
    )
    test.set_looking_at(test.cube_right[0][0])


class MemoryExplorer(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        add_two_cubes(self)


if __name__ == "__main__":
    import memory_filters as mf  # noqa

    M = MemoryExplorer()
    M.setUp()
    a = M.agent
    m = M.agent.memory
