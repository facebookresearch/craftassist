"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
# TODO correct model paths

import os
import sys
import logging
import faulthandler
import signal
import random
import re
import sentry_sdk
import numpy as np

# FIXME
import time
from multiprocessing import set_start_method

import inventory

import mc_memory
from dialogue_objects import GetMemoryHandler, PutMemoryHandler, Interpreter

BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(BASE_AGENT_ROOT)

import dashboard

if __name__ == "__main__":
    # this line has to go before any imports that contain @sio.on functions
    # or else, those @sio.on calls become no-ops
    print("starting dashboard...")
    dashboard.start()

from base_agent.loco_mc_agent import LocoMCAgent
from argument_parser import ArgumentParser

from agent import Agent as MCAgent

from low_level_perception import LowLevelMCPerception
import heuristic_perception

from mc_util import cluster_areas, hash_user, MCTime
from voxel_models.subcomponent_classifier import SubcomponentClassifierWrapper
from voxel_models.geoscorer import Geoscorer
from base_agent.nsp_dialogue_manager import NSPDialogueManager
import default_behaviors
import subprocess
import rotation

faulthandler.register(signal.SIGUSR1)

random.seed(0)
log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().handlers.clear()

sentry_sdk.init()  # enabled if SENTRY_DSN set in env

DEFAULT_BEHAVIOUR_TIMEOUT = 20
DEFAULT_FRAME = "SPEAKER"


class CraftAssistAgent(LocoMCAgent):
    default_frame = DEFAULT_FRAME
    coordinate_transforms = rotation

    def __init__(self, opts):
        super(CraftAssistAgent, self).__init__(opts)
        self.no_default_behavior = opts.no_default_behavior
        self.point_targets = []
        self.last_chat_time = 0
        # areas must be perceived at each step
        # List of tuple (XYZ, radius), each defines a cube
        self.areas_to_perceive = []
        self.add_self_memory_node()
        self.init_inventory()

    def init_inventory(self):
        self.inventory = inventory.Inventory()
        logging.info("Initialized agent inventory")

    def init_memory(self):
        self.memory = mc_memory.MCAgentMemory(
            db_file=os.environ.get("DB_FILE", ":memory:"),
            db_log_path="agent_memory.{}.log".format(self.name),
            agent_time=MCTime(self.get_world_time),
        )
        file_log_handler = logging.FileHandler("agent.{}.log".format(self.name))
        file_log_handler.setFormatter(log_formatter)
        logging.getLogger().addHandler(file_log_handler)
        logging.info("Initialized agent memory")

    def init_perception(self):
        self.perception_modules = {}
        self.perception_modules["low_level"] = LowLevelMCPerception(self)
        self.perception_modules["heuristic"] = heuristic_perception.PerceptionWrapper(self)
        # set up the SubComponentClassifier model
        if self.opts.semseg_model_path:
            self.perception_modules["semseg"] = SubcomponentClassifierWrapper(
                self, self.opts.semseg_model_path, self.opts.semseg_vocab_path
            )

        # set up the Geoscorer model
        self.geoscorer = (
            Geoscorer(merger_model_path=self.opts.geoscorer_model_path)
            if self.opts.geoscorer_model_path
            else None
        )

    def init_controller(self):
        dialogue_object_classes = {}
        dialogue_object_classes["interpreter"] = Interpreter
        dialogue_object_classes["get_memory"] = GetMemoryHandler
        dialogue_object_classes["put_memory"] = PutMemoryHandler
        self.dialogue_manager = NSPDialogueManager(self, dialogue_object_classes, self.opts)

    def perceive(self, force=False):
        self.areas_to_perceive = cluster_areas(self.areas_to_perceive)
        for v in self.perception_modules.values():
            v.perceive(force=force)
        self.areas_to_perceive = []

    def controller_step(self):
        """Process incoming chats and modify task stack"""
        raw_incoming_chats = self.get_incoming_chats()
        incoming_chats = []
        for raw_chat in raw_incoming_chats:
            match = re.search("^<([^>]+)> (.*)", raw_chat)
            if match is None:
                logging.info("Ignoring chat: {}".format(raw_chat))
                continue

            speaker, chat = match.group(1), match.group(2)
            speaker_hash = hash_user(speaker)
            logging.info("Incoming chat: ['{}' -> {}]".format(speaker_hash, chat))
            if chat.startswith("/"):
                continue
            incoming_chats.append((speaker, chat))
            self.memory.add_chat(self.memory.get_player_by_name(speaker).memid, chat)
        if incoming_chats:
            # force to get objects, speaker info
            self.perceive(force=True)
            # logging.info("Incoming chats: {}".format(raw_incoming_chats))

            # change this to memory.get_time() format?
            self.last_chat_time = time.time()
            # for now just process the first incoming chat
            self.dialogue_manager.step(incoming_chats[0])
        else:
            # Maybe add default task
            if not self.no_default_behavior:
                self.maybe_run_slow_defaults()
            self.dialogue_manager.step((None, ""))

    def maybe_run_slow_defaults(self):
        """Pick a default task task to run
        with a low probability"""
        if self.memory.task_stack_peek() or len(self.dialogue_manager.dialogue_stack) > 0:
            return

        # list of (prob, default function) pairs
        visible_defaults = [
            (0.001, default_behaviors.build_random_shape),
            (0.005, default_behaviors.come_to_player),
        ]
        # default behaviors of the agent not visible in the game
        invisible_defaults = []

        defaults = (
            visible_defaults + invisible_defaults
            if time.time() - self.last_chat_time > DEFAULT_BEHAVIOUR_TIMEOUT
            else invisible_defaults
        )

        defaults = [(p, f) for (p, f) in defaults if f not in self.memory.banned_default_behaviors]

        def noop(*args):
            pass

        defaults.append((1 - sum(p for p, _ in defaults), noop))  # noop with remaining prob

        # weighted random choice of functions
        p, fns = zip(*defaults)
        fn = np.random.choice(fns, p=p)
        if fn != noop:
            logging.info("Default behavior: {}".format(fn))
        fn(self)

    def get_time(self):
        # round to 100th of second, return as
        # n hundreth of seconds since agent init
        return self.memory.get_time()

    def get_world_time(self):
        # MC time is based on ticks, where 20 ticks happen every second.
        # There are 24000 ticks in a day, making Minecraft days exactly 20 minutes long.
        # The time of day in MC is based on the timestamp modulo 24000 (default).
        # 0 is sunrise, 6000 is noon, 12000 is sunset, and 18000 is midnight.
        return self.get_time_of_day()

    def safe_get_changed_blocks(self):
        blocks = self.cagent.get_changed_blocks()
        safe_blocks = []
        if len(self.point_targets) > 0:
            for point_target in self.point_targets:
                pt = point_target[0]
                for b in blocks:
                    x, y, z = b[0]
                    xok = x < pt[0] or x > pt[3]
                    yok = y < pt[1] or y > pt[4]
                    zok = z < pt[2] or z > pt[5]
                    if xok and yok and zok:
                        safe_blocks.append(b)
        else:
            safe_blocks = blocks
        return safe_blocks

    def point_at(self, target, sleep=None):
        """Bot pointing.

        Args:
            target: list of x1 y1 z1 x2 y2 z2, where:
                    x1 <= x2,
                    y1 <= y2,
                    z1 <= z2.
        """
        assert len(target) == 6
        self.send_chat("/point {} {} {} {} {} {}".format(*target))
        self.point_targets.append((target, time.time()))
        # sleep before the bot can take any actions
        # otherwise there might be bugs since the object is flashing
        # deal with this in the task...
        if sleep:
            time.sleep(sleep)

    def relative_head_pitch(self, angle):
        # warning: pitch is flipped!
        new_pitch = self.get_player().look.pitch - angle
        self.set_look(self.get_player().look.yaw, new_pitch)

    def send_chat(self, chat):
        logging.info("Sending chat: {}".format(chat))
        self.memory.add_chat(self.memory.self_memid, chat)
        return self.cagent.send_chat(chat)

    # TODO update client so we can just loop through these
    # TODO rename things a bit- some perceptual things are here,
    #      but under current abstraction should be in init_perception
    def init_physical_interfaces(self):
        # For testing agent without cuberite server
        if self.opts.port == -1:
            return
        logging.info("Attempting to connect to port {}".format(self.opts.port))
        self.cagent = MCAgent("localhost", self.opts.port, self.name)
        logging.info("Logged in to server")
        self.dig = self.cagent.dig
        self.drop_item_stack_in_hand = self.cagent.drop_item_stack_in_hand
        self.drop_item_in_hand = self.cagent.drop_item_in_hand
        self.drop_inventory_item_stack = self.cagent.drop_inventory_item_stack
        self.set_inventory_slot = self.cagent.set_inventory_slot
        self.get_player_inventory = self.cagent.get_player_inventory
        self.get_inventory_item_count = self.cagent.get_inventory_item_count
        self.get_inventory_items_counts = self.cagent.get_inventory_items_counts
        # defined above...
        # self.send_chat = self.cagent.send_chat
        self.set_held_item = self.cagent.set_held_item
        self.step_pos_x = self.cagent.step_pos_x
        self.step_neg_x = self.cagent.step_neg_x
        self.step_pos_z = self.cagent.step_pos_z
        self.step_neg_z = self.cagent.step_neg_z
        self.step_pos_y = self.cagent.step_pos_y
        self.step_neg_y = self.cagent.step_neg_y
        self.step_forward = self.cagent.step_forward
        self.look_at = self.cagent.look_at
        self.set_look = self.cagent.set_look
        self.turn_angle = self.cagent.turn_angle
        self.turn_left = self.cagent.turn_left
        self.turn_right = self.cagent.turn_right
        self.place_block = self.cagent.place_block
        self.use_entity = self.cagent.use_entity
        self.use_item = self.cagent.use_item
        self.use_item_on_block = self.cagent.use_item_on_block
        self.is_item_stack_on_ground = self.cagent.is_item_stack_on_ground
        self.craft = self.cagent.craft
        self.get_blocks = self.cagent.get_blocks
        self.get_local_blocks = self.cagent.get_local_blocks
        self.get_incoming_chats = self.cagent.get_incoming_chats
        self.get_player = self.cagent.get_player
        self.get_mobs = self.cagent.get_mobs
        self.get_other_players = self.cagent.get_other_players
        self.get_other_player_by_name = self.cagent.get_other_player_by_name
        self.get_vision = self.cagent.get_vision
        self.get_line_of_sight = self.cagent.get_line_of_sight
        self.get_player_line_of_sight = self.cagent.get_player_line_of_sight
        self.get_changed_blocks = self.cagent.get_changed_blocks
        self.get_item_stacks = self.cagent.get_item_stacks
        self.get_world_age = self.cagent.get_world_age
        self.get_time_of_day = self.cagent.get_time_of_day
        self.get_item_stack = self.cagent.get_item_stack

    def add_self_memory_node(self):
        # clean this up!  FIXME!!!!! put in base_agent_memory?
        # how/when to, memory is initialized before physical interfaces...
        try:
            p = self.get_player()
        except:  # this is for test/test_agent :(
            return
        self.memory._db_write(
            "INSERT INTO ReferenceObjects(uuid, eid, name, ref_type, x, y, z, pitch, yaw) VALUES (?,?,?,?,?,?,?,?,?)",
            self.memory.self_memid,
            p.entityId,
            p.name,
            "player",
            p.pos.x,
            p.pos.y,
            p.pos.z,
            p.look.pitch,
            p.look.yaw,
        )


if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    parser = ArgumentParser("Minecraft", base_path)
    opts = parser.parse()

    # set up stdout logging
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG if opts.verbose else logging.INFO)
    sh.setFormatter(log_formatter)
    logging.getLogger().addHandler(sh)
    logging.info("Info logging")
    logging.debug("Debug logging")

    # Check that models and datasets are up to date
    rc = subprocess.call([opts.verify_hash_script_path, "craftassist"])

    set_start_method("spawn", force=True)

    sa = CraftAssistAgent(opts)
    sa.start()
