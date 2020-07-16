"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import numpy as np
from typing import List

from util import XYZ, IDM, Block
from utils import Look, Pos, Item, Player
from base_agent.loco_mc_agent import LocoMCAgent
from mc_memory import MCAgentMemory
from mc_memory_nodes import VoxelObjectNode
from craftassist_agent import CraftAssistAgent
from base_agent.nsp_dialogue_manager import NSPDialogueManager
from dialogue_objects import GetMemoryHandler, PutMemoryHandler, Interpreter
from low_level_perception import LowLevelMCPerception
import heuristic_perception
from rotation import look_vec


class Opt:
    pass


class FakeCPPAction:
    NAME = "NULL"

    def __init__(self, agent):
        self.agent = agent

    def action(self, *args):
        pass

    def __call__(self, *args):
        if hasattr(self.agent, "recorder"):
            self.agent.recorder.record_action({"name": self.NAME, "args": list(args)})
        return self.action(*args)


class Dig(FakeCPPAction):
    NAME = "dig"

    def action(self, x, y, z):
        dug = self.agent.world.dig((x, y, z))
        if dug:
            self.agent._changed_blocks.append(((x, y, z), (0, 0)))
            return True
        else:
            return False


class SendChat(FakeCPPAction):
    NAME = "send_chat"

    def action(self, chat):
        logging.info("FakeAgent.send_chat: {}".format(chat))
        self.agent._outgoing_chats.append(chat)


class SetHeldItem(FakeCPPAction):
    NAME = "set_held_item"

    def action(self, arg):
        try:
            d, m = arg
            self.agent._held_item = (d, m)
        except TypeError:
            self.agent._held_item = (arg, 0)


class StepPosX(FakeCPPAction):
    NAME = "step_pos_x"

    def action(self):
        self.agent.pos += (1, 0, 0)


class StepNegX(FakeCPPAction):
    NAME = "step_neg_x"

    def action(self):
        self.agent.pos += (-1, 0, 0)


class StepPosZ(FakeCPPAction):
    NAME = "step_pos_z"

    def action(self):
        self.agent.pos += (0, 0, 1)


class StepNegZ(FakeCPPAction):
    NAME = "step_neg_z"

    def action(self):
        self.agent.pos += (0, 0, -1)


class StepPosY(FakeCPPAction):
    NAME = "step_pos_y"

    def action(self):
        self.agent.pos += (0, 1, 0)


class StepNegY(FakeCPPAction):
    NAME = "step_neg_y"

    def action(self):
        self.agent.pos += (0, -1, 0)


class StepForward(FakeCPPAction):
    NAME = "step_forward"

    def action(self):
        dx, dy, dz = self.agent._look_vec
        self.agent.pos += (dx, 0, dz)


class TurnAngle(FakeCPPAction):
    NAME = "turn_angle"

    def action(self, angle):
        if angle == 90:
            self.agent.turn_left()
        elif angle == -90:
            self.agent.turn_right()
        else:
            raise ValueError("bad angle={}".format(angle))


class TurnLeft(FakeCPPAction):
    NAME = "turn_left"

    def action(self):
        old_l = (self.agent._look_vec[0], self.agent._look_vec[1])
        idx = self.agent.CCW_LOOK_VECS.index(old_l)
        new_l = self.agent.CCW_LOOK_VECS[(idx + 1) % len(self.agent.CCW_LOOK_VECS)]
        self.agent._look_vec[0] = new_l[0]
        self.agent._look_vec[2] = new_l[2]


class TurnRight(FakeCPPAction):
    NAME = "turn_right"

    def action(self):
        old_l = (self.agent._look_vec[0], self.agent._look_vec[1])
        idx = self.agent.CCW_LOOK_VECS.index(old_l)
        new_l = self.agent.CCW_LOOK_VECS[(idx - 1) % len(self.agent.CCW_LOOK_VECS)]
        self.agent._look_vec[0] = new_l[0]
        self.agent._look_vec[2] = new_l[2]


class PlaceBlock(FakeCPPAction):
    NAME = "place_block"

    def action(self, x, y, z):
        block = ((x, y, z), self.agent._held_item)
        self.agent.world.place_block(block)
        self.agent._changed_blocks.append(block)
        return True


class LookAt(FakeCPPAction):
    NAME = "look_at"

    def action(self, x, y, z):
        raise NotImplementedError()


class SetLook(FakeCPPAction):
    NAME = "set_look"

    def action(self, yaw, pitch):
        a = look_vec(yaw, pitch)
        self._look_vec = [a[0], a[1], a[2]]


class Craft(FakeCPPAction):
    NAME = "craft"

    def action(self):
        raise NotImplementedError()


class FakeAgent(LocoMCAgent):
    CCW_LOOK_VECS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def __init__(self, world, opts=None, do_heuristic_perception=False):
        self.world = world
        self.chat_count = 0
        if not opts:
            opts = Opt()
            opts.nsp_model_dir = None
            opts.nsp_data_dir = None
            opts.nsp_embedding_path = None
            opts.model_base_path = None
            opts.QA_nsp_model_path = None
            opts.ground_truth_data_dir = ""
            opts.web_app = False
        super(FakeAgent, self).__init__(opts)
        self.do_heuristic_perception = do_heuristic_perception
        self.no_default_behavior = True
        self.last_task_memid = None
        pos = (0, 63, 0)
        if hasattr(self.world, "agent_data"):
            pos = self.world.agent_data["pos"]
        self.pos = np.array(pos, dtype="int")
        self.logical_form = None

        self._held_item: IDM = (0, 0)
        self._look_vec = (1, 0, 0)
        self._changed_blocks: List[Block] = []
        self._outgoing_chats: List[str] = []
        CraftAssistAgent.add_self_memory_node(self)

    def init_perception(self):
        self.geoscorer = None
        self.perception_modules = {}
        self.perception_modules["low_level"] = LowLevelMCPerception(self)
        self.perception_modules["heuristic"] = heuristic_perception.PerceptionWrapper(self)

    def init_physical_interfaces(self):
        self.dig = Dig(self)
        self.send_chat = SendChat(self)
        self.set_held_item = SetHeldItem(self)
        self.step_pos_x = StepPosX(self)
        self.step_neg_x = StepNegX(self)
        self.step_pos_z = StepPosZ(self)
        self.step_neg_z = StepNegZ(self)
        self.step_pos_y = StepPosY(self)
        self.step_neg_y = StepNegY(self)
        self.step_forward = StepForward(self)
        self.turn_angle = TurnAngle(self)
        self.turn_left = TurnLeft(self)
        self.turn_right = TurnRight(self)
        self.set_look = SetLook(self)
        self.place_block = PlaceBlock(self)

    def init_memory(self):
        self.memory = MCAgentMemory(load_minecraft_specs=False)  # don't load specs, it's slow

    def init_controller(self):
        dialogue_object_classes = {}
        dialogue_object_classes["interpreter"] = Interpreter
        dialogue_object_classes["get_memory"] = GetMemoryHandler
        dialogue_object_classes["put_memory"] = PutMemoryHandler
        self.dialogue_manager = NSPDialogueManager(self, dialogue_object_classes, self.opts)

    def set_logical_form(self, lf, chatstr, speaker):
        self.logical_form = {"logical_form": lf, "chatstr": chatstr, "speaker": speaker}

    def step(self):
        if hasattr(self.world, "step"):
            self.world.step()
        if hasattr(self, "recorder"):
            self.recorder.record_world()
        super().step()

    #### use the CraftassistAgent.controller_step()
    def controller_step(self):
        if self.logical_form is None:
            pass
            CraftAssistAgent.controller_step(self)
        else:  # logical form given directly:
            # clear the chat buffer
            self.get_incoming_chats()
            # use the logical form as given...
            d = self.logical_form["logical_form"]
            chatstr = self.logical_form["chatstr"]
            speaker_name = self.logical_form["speaker"]
            self.memory.add_chat(self.memory.get_player_by_name(speaker_name).memid, chatstr)
            obj = self.dialogue_manager.handle_logical_form(speaker_name, d, chatstr)
            if obj is not None:
                self.dialogue_manager.dialogue_stack.append(obj)
            self.logical_form = None

    def setup_test(self):
        self.task_steps_count = 0

    def clear_outgoing_chats(self):
        self._outgoing_chats.clear()

    def get_last_outgoing_chat(self):
        try:
            return self._outgoing_chats[-1]
        except IndexError:
            return None

    ########################
    ##  FAKE .PY METHODS  ##
    ########################

    def task_step(self):
        CraftAssistAgent.task_step(self, sleep_time=0)

    def point_at(*args):
        pass

    def perceive(self, force=False):
        self.perception_modules["low_level"].perceive(force=force)
        if self.do_heuristic_perception:
            self.perception_modules["heuristic"].perceive()

    ###################################
    ##  FAKE C++ PERCEPTION METHODS  ##
    ###################################

    def get_blocks(self, xa, xb, ya, yb, za, zb):
        return self.world.get_blocks(xa, xb, ya, yb, za, zb)

    def get_local_blocks(self, r):
        x, y, z = self.pos
        return self.get_blocks(x - r, x + r, y - r, y + r, z - r, z + r)

    def get_incoming_chats(self):
        c = self.chat_count
        self.chat_count = len(self.world.chat_log)
        return self.world.chat_log[c:].copy()

    def get_player(self):
        return Player(1, "fake_agent", Pos(*self.pos), self.get_look(), Item(*self._held_item))

    def get_mobs(self):
        return self.world.get_mobs()

    def get_item_stacks(self):
        return self.world.get_item_stacks()

    def get_other_players(self):
        return self.world.players.copy()

    def get_other_player_by_name(self):
        raise NotImplementedError()

    def get_vision(self):
        raise NotImplementedError()

    def get_line_of_sight(self):
        raise NotImplementedError()

    def get_look(self):
        pitch = -np.rad2deg(np.arcsin(self._look_vec[1]))
        yaw = -np.rad2deg(np.arctan2(self._look_vec[0], self._look_vec[2]))
        return Look(pitch, yaw)

    def get_player_line_of_sight(self, player_struct):
        if hasattr(self.world, "get_line_of_sight"):
            pos = (player_struct.pos.x, player_struct.pos.y, player_struct.pos.z)
            pitch = player_struct.look.pitch
            yaw = player_struct.look.yaw
            xsect = self.world.get_line_of_sight(pos, yaw, pitch)
            if xsect is not None:
                return Pos(*xsect)
        else:
            raise NotImplementedError()

    def get_changed_blocks(self) -> List[Block]:
        # need a better solution here
        r = self._changed_blocks.copy()
        self._changed_blocks.clear()
        return r

    def safe_get_changed_blocks(self) -> List[Block]:
        return self.get_changed_blocks()

    ######################################
    ## World setup
    ######################################

    def set_blocks(self, xyzbms: List[Block], origin: XYZ = (0, 0, 0)):
        """Change the state of the world, block by block,
        store in memory"""
        for xyz, idm in xyzbms:
            abs_xyz = tuple(np.array(xyz) + origin)
            self.perception_modules["low_level"].pending_agent_placed_blocks.add(abs_xyz)
            # TODO add force option so we don't need to make it as if agent placed
            self.perception_modules["low_level"].on_block_changed(abs_xyz, idm)
            self.world.place_block((abs_xyz, idm))

    def add_object(
        self, xyzbms: List[Block], origin: XYZ = (0, 0, 0), relations={}
    ) -> VoxelObjectNode:
        """Add an object to memory as if it was placed block by block

        Args:
        - xyzbms: a list of relative (xyz, idm)
        - origin: (x, y, z) of the corner

        Returns an VoxelObjectNode
        """
        self.set_blocks(xyzbms, origin)
        abs_xyz = tuple(np.array(xyzbms[0][0]) + origin)
        memid = self.memory.get_block_object_ids_by_xyz(abs_xyz)[0]
        for pred, obj in relations.items():
            self.memory.add_triple(subj=memid, pred_text=pred, obj_text=obj)
            # sooooorrry  FIXME? when we handle triples better in interpreter_helper
            if "has_" in pred:
                self.memory.tag(memid, obj)
        return self.memory.get_object_by_id(memid)

    ######################################
    ## visualization
    ######################################

    def draw_slice(self, h=None, r=5, c=None):
        if not h:
            h = self.pos[1]
        if c:
            c = [c[0], h, c[1]]
        else:
            c = [self.pos[0], h, self.pos[2]]
        C = self.world.to_world_coords(c)
        A = self.world.to_world_coords(self.pos)
        shifted_agent_pos = [A[0] - C[0] + r, A[2] - C[2] + r]
        npy = self.world.get_blocks(
            c[0] - r, c[0] + r, c[1], c[1], c[2] - r, c[2] + r, transpose=False
        )
        npy = npy[:, 0, :, 0]
        try:
            npy[shifted_agent_pos[0], shifted_agent_pos[1]] = 1024
        except:
            pass
        mobnums = {"rabbit": -1, "cow": -2, "pig": -3, "chicken": -4, "sheep": -5}
        nummobs = {-1: "rabbit", -2: "cow", -3: "pig", -4: "chicken", -5: "sheep"}
        for mob in self.world.mobs:
            # todo only in the plane?
            p = np.round(np.array(self.world.to_world_coords(mob.pos)))
            p = p - C
            try:
                npy[p[0] + r, p[1] + r] = mobnums[mob.mobname]
            except:
                pass
        mapslice = ""
        height = npy.shape[0]
        width = npy.shape[1]

        def xs(x):
            return x + int(self.pos[0]) - r

        def zs(z):
            return z + int(self.pos[2]) - r

        mapslice = mapslice + " " * (width + 2) * 3 + "\n"
        for i in reversed(range(height)):
            mapslice = mapslice + str(xs(i)).center(3)
            for j in range(width):
                if npy[i, j] > 0:
                    if npy[i, j] == 1024:
                        mapslice = mapslice + " A "
                    else:
                        mapslice = mapslice + str(npy[i, j]).center(3)
                elif npy[i, j] == 0:
                    mapslice = mapslice + " * "
                else:
                    npy[i, j] = mapslice + " " + nummobs[npy[i, j]][0] + " "
            mapslice = mapslice + "\n"
            mapslice = mapslice + "   "
            for j in range(width):
                mapslice = mapslice + " * "
            mapslice = mapslice + "\n"
        mapslice = mapslice + "   "
        for j in range(width):
            mapslice = mapslice + str(zs(j)).center(3)

        return mapslice
