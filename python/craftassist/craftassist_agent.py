import os
import sys

# python/ dir, for agent.so
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import faulthandler
import itertools
import logging
import numpy as np
import random
import re
import sentry_sdk
import signal
import time
from multiprocessing import set_start_method

from agent import Agent
from agent_connection import default_agent_name

# for visual recognition model
from vision_training.watcher import SubComponentClassifier
from freebuild.generator_3d import VoxelCNNGenerator

import build_utils as bu
import memory
import shapes
from util import to_block_pos, pos_to_np, TimingWarn

import default_behaviors
from ttad_model_dialogue_manager import TtadModelDialogueManager


faulthandler.register(signal.SIGUSR1)

random.seed(0)
log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().handlers.clear()

sentry_sdk.init()  # enabled if SENTRY_DSN set in env

DEFAULT_BEHAVIOUR_TIMEOUT = 20


class CraftAssistAgent(Agent):
    def __init__(
        self,
        host="localhost",
        port=25565,
        name=None,
        ttad_model_path=None,
        ttad_embeddings_path=None,
        ttad_grammar_path=None,
        center_voxel_model_path=None,
        voxel_model_gpu_id=-1,
        voxel_generator_model_path=None,
        get_objects_interval=20,
        draw_fn=None,
        no_default_behavior=False,
    ):
        logging.info("CraftAssistAgent.__init__ started")
        self.name = name or default_agent_name()
        self.no_default_behavior = no_default_behavior

        if ttad_model_path is None:
            ttad_model_path = os.path.join(os.path.dirname(__file__), "models/ttad/ttad.pth")
        if ttad_embeddings_path is None:
            ttad_embeddings_path = os.path.join(
                os.path.dirname(__file__), "models/ttad/ttad_ft_embeds.pth"
            )
        if ttad_grammar_path is None:
            ttad_grammar_path = os.path.join(
                os.path.dirname(__file__), "models/ttad/dialogue_grammar.json"
            )
        if center_voxel_model_path is None:
            center_voxel_model_path = os.path.join(
                os.path.dirname(__file__), "models/voxel-LocalVoxelCNN-t20.pth"
            )
            self.watcher = SubComponentClassifier(
                center_voxel_model_path, voxel_model_opts={"gpu_id": voxel_model_gpu_id}
            )
        else:
            self.watcher = None

        if voxel_generator_model_path is None:
            voxel_generator_model_path = os.path.join(
                os.path.dirname(__file__), "models/voxel-generator3d.pth"
            )
            self.generator = VoxelCNNGenerator(voxel_cnn_gen_dirs=voxel_generator_model_path)
        else:
            self.generator = None

        self.memory = memory.AgentMemory(
            db_file=os.environ.get("DB_FILE", ":memory:"),
            db_log_path="agent_memory.{}.log".format(self.name),
        )
        logging.info("Initialized AgentMemory")

        self.dialogue_manager = TtadModelDialogueManager(
            self, ttad_model_path, ttad_embeddings_path, ttad_grammar_path
        )
        logging.info("Initialized DialogueManager")

        # Log to file
        fh = logging.FileHandler("agent.{}.log".format(self.name))
        fh.setFormatter(log_formatter)
        fh.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(fh)

        # Login to server
        super().__init__(host, port, self.name)
        logging.info("Logged in to server")

        # Wrap C++ agent methods
        self._cpp_send_chat = self.send_chat
        self.send_chat = self._send_chat
        self.last_chat_time = 0

        self.get_objects_interval = get_objects_interval
        self.uncaught_error_count = 0
        self.last_task_memid = None

    def start(self):
        logging.info("CraftAssistAgent.start() called")
        if self.watcher:
            self.watcher.start()
        for self.count in itertools.count():  # count forever
            try:
                if self.count == 0:
                    logging.info("First top-level step()")
                self.step()

            except Exception as e:
                logging.exception(
                    "Default handler caught exception, db_log_idx={}".format(
                        self.memory.get_db_log_idx()
                    )
                )
                self.send_chat("Oops! I got confused and wasn't able to complete my last task :(")
                sentry_sdk.capture_exception(e)
                self.memory.task_stack_clear()
                self.dialogue_manager.dialogue_stack.clear()
                self.uncaught_error_count += 1
                if self.uncaught_error_count >= 100:
                    sys.exit(1)

    def step(self):
        self.pos = to_block_pos(pos_to_np(self.get_player().pos))

        # Update memory with current world state
        # Removed get_objects call due to very slow updates on non-flatworlds
        # with TimingWarn(2):
        #     self.get_objects()
        with TimingWarn(2):
            self.memory.update(self)

        # Process incoming chats
        self.dialogue_step()

        # Step topmost task on stack
        self.task_step()

    def task_step(self, sleep_time=0.25):
        # Clean finished tasks
        while (
            self.memory.task_stack_peek() and self.memory.task_stack_peek().task.check_finished()
        ):
            self.memory.task_stack_pop()

        # Maybe add default task
        if not self.no_default_behavior:
            self.maybe_run_slow_defaults()

        # If nothing to do, wait a moment
        if self.memory.task_stack_peek() is None:
            time.sleep(sleep_time)
            return

        # If something to do, step the topmost task
        task_mem = self.memory.task_stack_peek()
        if task_mem.memid != self.last_task_memid:
            logging.info("Starting task {}".format(task_mem.task))
            self.last_task_memid = task_mem.memid
        task_mem.task.step(self)
        self.memory.task_stack_update_task(task_mem.memid, task_mem.task)

    def get_objects(self, force=False):
        """
        Get both block objects and component objects and put them
        in memory
        """
        if not force and (
            self.count % self.get_objects_interval != 0
            or self.memory.task_stack_peek() is not None
        ):
            return

        block_objs_for_vision = []
        for obj in bu.all_nearby_objects(self.get_blocks, self.pos):
            memory.BlockObjectNode.create(self.memory, obj)
            # If any xyz of obj is has not been labeled
            if any([(not self.memory.get_component_object_ids_by_xyz(xyz)) for xyz, _ in obj]):
                block_objs_for_vision.append(obj)

        if self.watcher is None:
            return

        for obj in block_objs_for_vision:
            self.watcher.block_objs_q.put(obj)

        # everytime we try to retrieve as many recognition results as possible
        while not self.watcher.loc2labels_q.empty():
            loc2labels, obj = self.watcher.loc2labels_q.get()
            loc2ids = dict(obj)
            label2blocks = {}

            def contaminated(blocks):
                """
                Check if blocks are still consistent with the current world
                """
                mx, Mx, my, My, mz, Mz = shapes.get_bounds(blocks)
                yzxb = self.get_blocks(mx, Mx, my, My, mz, Mz)
                for b, _ in blocks:
                    x, y, z = b
                    if loc2ids[b][0] != yzxb[y - my, z - mz, x - mx, 0]:
                        return True
                return False

            for loc, labels in loc2labels.items():
                b = (loc, loc2ids[loc])
                for l in labels:
                    if l in label2blocks:
                        label2blocks[l].append(b)
                    else:
                        label2blocks[l] = [b]
            for l, blocks in label2blocks.items():
                ## if the blocks are contaminated we just ignore
                if not contaminated(blocks):
                    memory.ComponentObjectNode.create(self.memory, blocks, [l])

    def maybe_run_slow_defaults(self):
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

    def dialogue_step(self):
        """Process incoming chats and modify task stack"""
        raw_incoming_chats = self.get_incoming_chats()
        if raw_incoming_chats:
            # force to get objects
            self.get_objects(force=True)
            logging.info("Incoming chats: {}".format(raw_incoming_chats))
        incoming_chats = []
        for raw_chat in raw_incoming_chats:
            match = re.search("^<([^>]+)> (.*)", raw_chat)
            if match is None:
                logging.info("Ignoring chat: {}".format(raw_chat))
                continue
            speaker, chat = match.group(1), match.group(2)
            if chat.startswith("/"):
                continue
            incoming_chats.append((speaker, chat))
            self.memory.add_chat(self.memory.get_player_by_name(speaker).memid, chat)

        if len(incoming_chats) > 0:
            self.last_chat_time = time.time()
            # for now just process the first incoming chat:
            self.dialogue_manager.step(incoming_chats[0])
        else:
            self.dialogue_manager.step((None, ""))

    def point_at(self, target):
        """Bot pointing.

        Args:
            target: list of x1 y1 z1 x2 y2 z2, where:
                    x1 <= x2,
                    y1 <= y2,
                    z1 <= z2.
        """
        assert len(target) == 6
        self.send_chat("/point {} {} {} {} {} {}".format(*target))
        # sleep before the bot can take any actions
        # otherwise there might be bugs since the object is flashing
        time.sleep(4)

    def _send_chat(self, chat: str):
        logging.info("Sending chat: {}".format(chat))
        self.memory.add_chat(self.memory.self_memid, chat)
        return self._cpp_send_chat(chat)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--center_voxel_model_path", type=str, help="path to center voxel model")
    parser.add_argument("--gpu_id", type=int, default=-1, help="GPU id (-1 for cpu)")
    parser.add_argument("--ttad_model_path", help="path to listener model")
    parser.add_argument("--draw_vis", action="store_true", help="use visdom to draw agent vision")
    parser.add_argument(
        "--no_default_behavior",
        action="store_true",
        help="do not perform default behaviors when idle",
    )
    parser.add_argument("--name", help="Agent login name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    opts = parser.parse_args()

    # set up stdout logging
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG if opts.verbose else logging.INFO)
    sh.setFormatter(log_formatter)
    logging.getLogger().addHandler(sh)
    logging.info("Info logging")
    logging.debug("Debug logging")

    draw_fn = None
    if opts.draw_vis:
        import train_cnn

        draw_fn = train_cnn.draw_img

    set_start_method("spawn", force=True)

    sa = CraftAssistAgent(
        ttad_model_path=opts.ttad_model_path,
        center_voxel_model_path=opts.center_voxel_model_path,
        voxel_model_gpu_id=opts.gpu_id,
        draw_fn=draw_fn,
        no_default_behavior=opts.no_default_behavior,
        name=opts.name,
    )
    sa.start()
