"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest
from unittest.mock import Mock

import perception
import shapes
from base_craftassist_test_case import BaseCraftassistTestCase
from dialogue_objects.interpreter_helper import NextDialogueStep
from typing import List
from util import Block, strip_idmeta, euclid_dist


class TwoCubesInterpreterTest(BaseCraftassistTestCase):
    """A basic general-purpose test suite in a world which begins with two cubes.

    N.B. by default, the agent is looking at cube_right
    """

    def setUp(self):
        super().setUp()
        self.cube_right: List[Block] = list(
            self.add_object(shapes.cube(bid=(42, 0)), (9, 63, 4)).blocks.items()
        )
        self.cube_left: List[Block] = list(
            self.add_object(shapes.cube(), (9, 63, 10)).blocks.items()
        )

        self.set_looking_at(self.cube_right[0][0])

    def test_noop(self):
        d = {"dialogue_type": "NOOP"}
        changes = self.handle_action_dict(d)
        self.assertEqual(len(changes), 0)

    def test_destroy_that(self):
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": self.possible_actions["destroy_speaker_look"],
        }
        self.handle_action_dict(d)

        # Check that cube_right is destroyed
        self.assertEqual(
            set(self.get_blocks(strip_idmeta(self.cube_right)).values()), set([(0, 0)])
        )

    def test_copy_that(self):
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": self.possible_actions["copy_speaker_look_to_agent_pos"],
        }
        changes = self.handle_action_dict(d)

        # check that another gold cube was built
        self.assert_schematics_equal(list(changes.items()), self.cube_right)

    def test_build_small_sphere(self):
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": self.possible_actions["build_small_sphere"],
        }
        changes = self.handle_action_dict(d)

        # check that a small object was built
        self.assertGreater(len(changes), 0)
        self.assertLess(len(changes), 30)

    def test_build_1x1x1_cube(self):
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": self.possible_actions["build_1x1x1_cube"],
        }
        changes = self.handle_action_dict(d)

        # check that a single block will be built
        self.assertEqual(len(changes), 1)

    def test_move_coordinates(self):
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": {
                "action_type": "MOVE",
                "location": {"location_type": "COORDINATES", "coordinates": "-7 63 -8"},
            },
        }
        self.handle_action_dict(d)

        # check that agent moved
        self.assertLessEqual(euclid_dist(self.agent.pos, (-7, 63, -8)), 1)

    def test_move_here(self):
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": self.possible_actions["move_speaker_pos"],
        }
        self.handle_action_dict(d)

        # check that agent moved
        self.assertLessEqual(euclid_dist(self.agent.pos, self.get_speaker_pos()), 1)

    def test_build_diamond(self):
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": self.possible_actions["build_diamond"],
        }
        changes = self.handle_action_dict(d)

        # check that a Build was added with a single diamond block
        self.assertEqual(len(changes), 1)
        self.assertEqual(list(changes.values())[0], (57, 0))

    def test_build_gold_cube(self):
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": self.possible_actions["build_gold_cube"],
        }
        changes = self.handle_action_dict(d)

        # check that a Build was added with a gold blocks
        self.assertGreater(len(changes), 0)
        self.assertEqual(set(changes.values()), set([(41, 0)]))

    def test_fill_all_holes_no_holes(self):
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": self.possible_actions["fill_all_holes_speaker_look"],
        }
        perception.get_all_nearby_holes = Mock(return_value=[])  # no holes
        self.handle_action_dict(d)

    def test_go_to_the_tree(self):
        d = {"dialogue_type": "HUMAN_GIVE_COMMAND", "action": self.possible_actions["go_to_tree"]}
        try:
            self.handle_action_dict(d)
        except NextDialogueStep:
            pass

    def test_build_has_base(self):
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": {
                "action_type": "BUILD",
                "schematic": {
                    "has_block_type": "stone",
                    "has_name": "rectangle",
                    "has_height": "9",
                    "has_base": "9",  # has_base doesn't belong in "rectangle"
                },
            },
        }
        self.handle_action_dict(d)

    def test_build_square_has_height(self):
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": self.possible_actions["build_square_height_1"],
        }
        changes = self.handle_action_dict(d)
        ys = set([y for (x, y, z) in changes.keys()])
        self.assertEqual(len(ys), 1)  # height 1

    def test_action_sequence_order(self):
        target1 = (3, 63, 2)
        target2 = (7, 63, 7)
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action_sequence": [
                {
                    "action_type": "MOVE",
                    "location": {"location_type": "COORDINATES", "coordinates": str(target1)},
                },
                {
                    "action_type": "MOVE",
                    "location": {"location_type": "COORDINATES", "coordinates": str(target2)},
                },
            ],
        }

        self.handle_action_dict(d)
        self.assertLessEqual(euclid_dist(self.agent.pos, target2), 1)

    def test_stop(self):
        # start moving
        target = (20, 63, 20)
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": {
                "action_type": "MOVE",
                "location": {"location_type": "COORDINATES", "coordinates": str(target)},
            },
        }
        self.handle_action_dict(d, max_steps=5)

        # stop
        d = {"dialogue_type": "HUMAN_GIVE_COMMAND", "action": self.possible_actions["stop"]}
        self.handle_action_dict(d)

        # assert that move did not complete
        self.assertGreater(euclid_dist(self.agent.pos, target), 1)

    def test_build_sphere_move_here(self):
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action_sequence": [
                self.possible_actions["build_small_sphere"],
                self.possible_actions["move_speaker_pos"],
            ],
        }
        changes = self.handle_action_dict(d)

        # check that a small object was built
        self.assertGreater(len(changes), 0)
        self.assertLess(len(changes), 30)

        # check that agent moved
        self.assertLessEqual(euclid_dist(self.agent.pos, self.get_speaker_pos()), 1)

    def test_copy_that_and_build_cube(self):
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action_sequence": [
                self.possible_actions["copy_speaker_look_to_agent_pos"],
                self.possible_actions["build_1x1x1_cube"],
            ],
        }
        changes = self.handle_action_dict(d)

        # check that the cube_right is rebuilt and an additional block is built
        self.assertEqual(len(changes), len(self.cube_right) + 1)


class FillTest(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        self.hole_poss = [(x, 62, z) for x in (8, 9) for z in (10, 11)]
        self.set_blocks([(pos, (0, 0)) for pos in self.hole_poss])
        self.set_looking_at(self.hole_poss[0])
        self.assertEqual(set(self.get_blocks(self.hole_poss).values()), set([(0, 0)]))

    def test_fill_that(self):
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": self.possible_actions["fill_speaker_look"],
        }
        self.handle_action_dict(d)

        # Make sure hole is filled
        self.assertEqual(set(self.get_blocks(self.hole_poss).values()), set([(2, 0)]))

    def test_fill_with_block_type(self):
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": self.possible_actions["fill_speaker_look_gold"],
        }
        self.handle_action_dict(d)

        # Make sure hole is filled with gold
        self.assertEqual(set(self.get_blocks(self.hole_poss).values()), set([(41, 0)]))


if __name__ == "__main__":
    unittest.main()
