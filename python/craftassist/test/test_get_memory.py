"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest

import shapes
from base_craftassist_test_case import BaseCraftassistTestCase


class GetMemoryTestCase(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        self.cube_right = self.add_object(shapes.cube(bid=(42, 0)), (9, 63, 4))
        self.cube_left = self.add_object(shapes.cube(), (9, 63, 10))
        self.set_looking_at(list(self.cube_right.blocks.keys())[0])

    def test_get_name(self):
        # set the name
        name = "fluffball"
        self.agent.memory.add_triple(self.cube_right.memid, "has_name", name)

        # get the name
        d = {
            "dialogue_type": "GET_MEMORY",
            "filters": {
                "type": "REFERENCE_OBJECT",
                "reference_object": {"location": {"location_type": "SPEAKER_LOOK"}},
            },
            "answer_type": "TAG",
            "tag_name": "has_name",
        }
        self.handle_logical_form(d, stop_on_chat=True)

        # check that proper chat was sent
        self.assertIn(name, self.last_outgoing_chat())

    def test_what_are_you_doing(self):
        # start building a cube
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": {
                "action_type": "BUILD",
                "schematic": {"has_name": "cube", "has_size": "small"},
            },
        }
        self.handle_logical_form(d, max_steps=5)

        # what are you doing?
        d = {
            "dialogue_type": "GET_MEMORY",
            "filters": {"type": "ACTION"},
            "answer_type": "TAG",
            "tag_name": "action_name",
        }
        self.handle_logical_form(d, stop_on_chat=True)

        # check that proper chat was sent
        self.assertIn("building", self.last_outgoing_chat())

    def test_what_are_you_building(self):
        # start building a cube
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": {
                "action_type": "BUILD",
                "schematic": {"has_name": "cube", "has_size": "small"},
            },
        }
        self.handle_logical_form(d, max_steps=5)

        # what are you building
        d = {
            "dialogue_type": "GET_MEMORY",
            "filters": {"type": "ACTION", "action_type": "BUILD"},
            "answer_type": "TAG",
            "tag_name": "action_reference_object_name",
        }
        self.handle_logical_form(d, stop_on_chat=True)

        # check that proper chat was sent
        self.assertIn("cube", self.last_outgoing_chat())

    def test_where_are_you_going(self):
        # start moving
        target = (42, 65, 0)
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": {
                "action_type": "MOVE",
                "location": {
                    "location_type": "COORDINATES",
                    "coordinates": " ".join(map(str, target)),
                },
            },
        }
        self.handle_logical_form(d, max_steps=3)

        # where are you going?
        d = {
            "dialogue_type": "GET_MEMORY",
            "filters": {"type": "ACTION", "action_type": "MOVE"},
            "answer_type": "TAG",
            "tag_name": "move_target",
        }
        self.handle_logical_form(d, stop_on_chat=True)

        # check that proper chat was sent
        for x in target:
            self.assertIn(str(x), self.last_outgoing_chat())

    def test_where_are_you(self):
        # move to origin
        target = (0, 63, 0)
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": {
                "action_type": "MOVE",
                "location": {
                    "location_type": "COORDINATES",
                    "coordinates": " ".join(map(str, target)),
                },
            },
        }
        self.handle_logical_form(d)

        # where are you?
        d = {
            "dialogue_type": "GET_MEMORY",
            "filters": {"type": "AGENT"},
            "answer_type": "TAG",
            "tag_name": "location",
        }
        self.handle_logical_form(d)

        # check that proper chat was sent
        for x in target:
            self.assertIn(str(x), self.last_outgoing_chat())


if __name__ == "__main__":
    unittest.main()
