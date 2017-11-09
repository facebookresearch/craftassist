import unittest

import shapes
from dialogue_objects import AwaitResponse
from base_craftassist_test_case import BaseCraftassistTestCase


class UndoTest(BaseCraftassistTestCase):
    def test_undo_destroy(self):
        tag = "fluffy"

        # Build something
        obj = self.add_object(shapes.cube(bid=(41, 0)), (0, 63, 0))
        self.set_looking_at(list(obj.blocks.keys())[0])

        # Tag it
        d = {
            "dialogue_type": "PUT_MEMORY",
            "filters": {"reference_object": {"coref_resolve": "that"}},
            "upsert": {"memory_data": {"memory_type": "TRIPLE", "has_tag": "fluffy"}},
        }
        self.handle_action_dict(d)
        self.assertIn(tag, obj.get_tags())

        # Destroy it
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": {
                "action_type": "DESTROY",
                "reference_object": {"location": {"location_type": "SPEAKER_LOOK"}},
            },
        }
        self.handle_action_dict(d)
        self.assertIsNone(self.memory.get_block_object_by_xyz(list(obj.blocks.keys())[0]))

        # Undo destroy (will ask confirmation)
        d = {"dialogue_type": "HUMAN_GIVE_COMMAND", "action": {"action_type": "UNDO"}}
        self.handle_action_dict(d)
        self.assertIsInstance(self.dialogue_manager.dialogue_stack.peek(), AwaitResponse)

        # confirm undo
        self.add_incoming_chat("yes")
        self.flush()

        # Check that block object has tag
        newobj = self.memory.get_block_object_by_xyz(list(obj.blocks.keys())[0])
        self.assertIsNotNone(newobj)
        self.assertIn(tag, newobj.get_tags())


if __name__ == "__main__":
    unittest.main()
