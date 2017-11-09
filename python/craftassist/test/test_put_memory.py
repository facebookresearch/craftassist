import unittest

import shapes
from base_craftassist_test_case import BaseCraftassistTestCase


class PutMemoryTestCase(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        self.cube_right = self.add_object(shapes.cube(bid=(42, 0)), (9, 63, 4))
        self.cube_left = self.add_object(shapes.cube(), (9, 63, 10))
        self.set_looking_at(list(self.cube_right.blocks.keys())[0])

    def test_good_job(self):
        d = {
            "dialogue_type": "PUT_MEMORY",
            "upsert": {"memory_data": {"memory_type": "REWARD", "reward_value": "POSITIVE"}},
        }
        self.handle_action_dict(d)

    def test_tag(self):
        d = {
            "dialogue_type": "PUT_MEMORY",
            "filters": {"reference_object": {"location": {"location_type": "SPEAKER_LOOK"}}},
            "upsert": {"memory_data": {"memory_type": "TRIPLE", "has_tag": "fluffy"}},
        }
        self.handle_action_dict(d)

        # destroy it
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": {"action_type": "DESTROY", "reference_object": {"has_tag": "fluffy"}},
        }
        changes = self.handle_action_dict(d, answer="yes")

        # ensure it was destroyed
        self.assertEqual(changes, {k: (0, 0) for k in self.cube_right.blocks.keys()})

    def test_tag_and_build(self):
        tag = "fluffy"
        d = {
            "dialogue_type": "PUT_MEMORY",
            "filters": {"reference_object": {"location": {"location_type": "SPEAKER_LOOK"}}},
            "upsert": {"memory_data": {"memory_type": "TRIPLE", "has_tag": tag}},
        }
        self.handle_action_dict(d)

        # build a fluffy
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": {
                "action_type": "BUILD",
                "schematic": {"has_name": tag},
                "location": {"location_type": "AGENT_POS"},
            },
        }
        changes = self.handle_action_dict(d, answer="yes")

        self.assert_schematics_equal(changes.items(), self.cube_right.blocks.items())


if __name__ == "__main__":
    unittest.main()
