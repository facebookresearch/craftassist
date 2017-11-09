import gzip
import logging
import numpy as np
import os
import pickle
import random
import sqlite3
import time
import uuid
from collections import Counter
from itertools import zip_longest
from typing import cast, Optional, List, Tuple, Dict, Sequence, Union

from block_data import BORING_BLOCKS
from build_utils import npy_to_blocks_list
from entities import MOBS_BY_ID
import minecraft_specs
import util
import dance
from util import XYZ, IDM, Block
from tasks import Task

SCHEMA = os.path.join(os.path.dirname(__file__), "memory_schema.sql")

# TODO "snapshot" memory type  (giving a what mob/object/player looked like at a fixed timestamp)
# TODO when a memory is removed, its last state should be snapshotted to prevent tag weirdness


class AgentMemory:
    def __init__(
        self, db_file=":memory:", db_log_path=None, task_db={}, load_minecraft_specs=True
    ):
        self.db = sqlite3.connect(db_file)

        self.task_db = task_db

        self.banned_default_behaviors = []  # FIXME: move into triple store?
        self.other_players = {}
        self.pending_agent_placed_blocks = set()
        self.updateable_mems = []
        self._safe_pickle_saved_attrs = {}

        if db_log_path:
            self._db_log_file = gzip.open(db_log_path + ".gz", "w")
            self._db_log_idx = 0

        with open(SCHEMA, "r") as f:
            self._db_script(f.read())

        self._load_schematics(load_minecraft_specs)

        # self.memids = {}  # TODO: key is memid, value is sql table name or memory dict name
        self.all_tables = [
            c[0] for c in self._db_read("SELECT name FROM sqlite_master WHERE type='table';")
        ]

        self.dances = {}
        dance.add_default_dances(self)

        # create a "self" memory to reference in Triples
        self.self_memid = "0" * len(uuid.uuid4().hex)
        self._db_write("INSERT INTO Memories VALUES (?,?,?)", self.self_memid, "Memories", 0)
        self.tag(self.self_memid, "_agent")
        self.tag(self.self_memid, "_self")

    def __del__(self):
        if getattr(self, "_db_log_file", None):
            self._db_log_file.close()

    # TODO list of all "updatable" mems, do a mem.update() ?
    def update(self, agent):
        for mob in agent.get_mobs():
            self.set_mob_position(mob)

        for (xyz, idm) in agent.get_changed_blocks():
            self.on_block_changed(xyz, idm)

        self.update_other_players(agent.get_other_players())

    ########################
    ### Workspace memory ###
    ########################

    def update_recent_entities(self, mems=[]):
        logging.info("update_recent_entities {}".format(mems))
        for mem in mems:
            mem.update_recently_used()

    def get_recent_entities(self, memtype, time_window=120) -> List["MemoryNode"]:
        r = self._db_read(
            """SELECT uuid
            FROM Memories
            WHERE tabl=? AND workspace_updated_time > ?
            ORDER BY workspace_updated_time DESC""",
            memtype,
            time.time() - time_window,
        )
        return [self.get_mem_by_id(memid, memtype) for memid, in r]

    ###############
    ### General ###
    ###############

    def get_memid_table(self, memid: str) -> str:
        r, = self._db_read_one("SELECT tabl FROM Memories WHERE uuid=?", memid)
        return r

    def get_mem_by_id(self, memid: str, table: str = None) -> "MemoryNode":
        if table is None:
            table = self.get_memid_table(memid)

        def dummy_s(g):  # sorry :(
            return lambda s, y: g(y)

        if table is None:
            return
        from_id = {
            "BlockObjects": dummy_s(self.get_object_by_id),
            "ComponentObjects": dummy_s(self.get_component_object_by_id),
            "Schematics": dummy_s(self.get_schematic_by_id),
            "Chats": dummy_s(self.get_chat_by_id),
            "Mobs": dummy_s(self.get_mob_by_id),
            "Tasks": dummy_s(self.get_task_by_id),
            "Players": lambda s, memid: PlayerNode(s, memid),
            "Self": lambda s, memid: MemoryNode(s, memid),
            "Triples": lambda s, memid: MemoryNode(s, memid),
            "Locations": lambda s, memid: MemoryNode(s, memid),
            "Dances": lambda s, memid: MemoryNode(s, memid),
        }
        return from_id[table](self, memid)

    def get_all_tagged_mems(self, tag: str) -> List["MemoryNode"]:
        memids = self.get_memids_by_tag(tag)
        return [self.get_mem_by_id(memid) for memid in memids]

    def check_memid_exists(self, memid: str, table: str) -> bool:
        return bool(self._db_read_one("SELECT * FROM {} WHERE uuid=?".format(table), memid))

    def forget(self, memid: str, hard=True):
        T = self.get_memid_table(memid)
        if not hard:
            self.add_triple(memid, "has_tag", "_forgotten")
        else:
            if T is not None:
                self._db_write("DELETE FROM {} WHERE uuid=?".format(T), memid)
                # TODO this less brutally.  might want to remember some
                # triples where the subject or object has been removed
                # eventually we might have second-order relations etc, this could set
                # off a chain reaction
                self.remove_memid_triple(memid, role="both")

    ###############
    ### Blocks  ###
    ###############

    def on_block_changed(self, xyz: XYZ, idm: IDM):
        if idm[0] == 0:
            # block removed
            tables = ["BlockObjects", "ComponentObjects"]
            for table in tables:
                memids = self.get_object_ids_by_xyz(xyz, table)
                if not memids:
                    return
                assert len(memids) == 1
                self._db_write("DELETE FROM {} WHERE x=? AND y=? AND z=?".format(table), *xyz)
                if not self.check_memid_exists(memids[0], table):
                    # object is gone now
                    self.remove_memid_triple(memids[0])

        else:
            # block added
            interesting, player_placed = self._is_placed_block_interesting(xyz, idm[0])
            if not interesting:
                return

            adjacent_memids = set.union(
                *[set(self.get_block_object_ids_by_xyz(a)) for a in util.diag_adjacent(xyz)]
            )

            if len(adjacent_memids) == 0:
                # new block object
                BlockObjectNode.create(self, [(xyz, idm)])
            elif len(adjacent_memids) == 1:
                # update block object
                memid = adjacent_memids.pop()
                self._upsert_block((xyz, idm), memid, "BlockObjects", player_placed)
            else:
                chosen_memid = list(adjacent_memids)[0]

                # merge tags
                where = " OR ".join(["subj=?"] * len(adjacent_memids))
                self._db_write(
                    "UPDATE Triples SET subj=? WHERE " + where, chosen_memid, *adjacent_memids
                )

                # merge multiple block objects (will delete old ones)
                where = " OR ".join(["uuid=?"] * len(adjacent_memids))
                self._db_write(
                    "UPDATE BlockObjects SET uuid=? WHERE " + where, chosen_memid, *adjacent_memids
                )

                # insert new block
                self._upsert_block((xyz, idm), chosen_memid, "BlockObjects", player_placed)

    def _upsert_block(self, block: Block, memid: str, table: str, player_placed: bool = False):
        (x, y, z), (b, m) = block
        if self._db_read_one("SELECT * FROM {} WHERE x=? AND y=? AND z=?".format(table), x, y, z):
            self._db_write(
                """
                    UPDATE {}
                    SET uuid=?, bid=?, meta=?, updated=?, player_placed=?
                    WHERE x=? AND y=? AND z=?
                    """.format(
                    table
                ),
                memid,
                b,
                m,
                time.time(),
                player_placed,
                x,
                y,
                z,
            )
        else:
            self._db_write(
                """
                    INSERT INTO {}(uuid, x, y, z, bid, meta, updated, player_placed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """.format(
                    table
                ),
                memid,
                x,
                y,
                z,
                b,
                m,
                time.time(),
                player_placed,
            )

    def get_object_by_id(self, memid: str, table="BlockObjects") -> "ObjectNode":
        r = self._db_read(
            "SELECT x, y, z, bid, meta FROM {} WHERE uuid=? ORDER BY updated".format(table), memid
        )
        r = list(r)
        assert r, memid
        last_placed = tuple(r[-1][0:3])  # (x, y, z) for last block
        if table == "BlockObjects":
            return BlockObjectNode(self, memid, cast(XYZ, last_placed))
        elif table == "ComponentObjects":
            return ComponentObjectNode(self, memid, cast(XYZ, last_placed))
        else:
            raise ValueError("Bad table={}".format(table))

    def get_object_ids_by_xyz(self, xyz: XYZ, table: str) -> List[str]:
        r = self._db_read(
            "SELECT DISTINCT(uuid) FROM {} WHERE x=? AND y=? AND z=?".format(table), *xyz
        )
        return [x for (x,) in r]

    #################
    ###  Triples  ###
    #################

    def add_triple(self, subj: str, pred: str, obj: str, confidence=1.0):
        # FIXME: remove this if seen after April 18, 2019
        if obj == "nothing":
            return

        memid = uuid.uuid4().hex
        self._db_write(
            "INSERT INTO Triples VALUES (?, ?, ?, ?, ?)", memid, subj, pred, obj, confidence
        )

    def tag(self, subj_memid: str, tag: str):
        self.add_triple(subj_memid, "has_tag", tag)

    def untag(self, subj_memid: str, tag: str):
        self._db_write(
            'DELETE FROM Triples WHERE subj=? AND pred="has_tag" AND obj=?', subj_memid, tag
        )

    def get_memids_by_tag(self, tag: str) -> List[str]:
        r = self._db_read('SELECT DISTINCT(subj) FROM Triples WHERE pred="has_tag" AND obj=?', tag)
        return [x for (x,) in r]

    # TODO rename me get_tags_by_subj
    def get_tags_by_memid(self, subj_memid: str) -> List[str]:
        r = self._db_read(
            'SELECT DISTINCT(obj) FROM Triples WHERE pred="has_tag" AND subj=?', subj_memid
        )
        return [x for (x,) in r]

    # TODO rename me get_triple_memids
    def get_triples(
        self, subj: str = None, pred: str = None, obj: str = None
    ) -> List[Tuple[str, str, str]]:
        # subj should be a memid,
        # and at least one of the three should not be Null
        assert any([subj, obj, pred])
        pairs = [("subj", subj), ("pred", pred), ("obj", obj)]
        args = [x[1] for x in pairs if x[1] is not None]
        where = [x[0] + "=?" for x in pairs if x[1] is not None]
        if len(where) == 1:
            where_clause = where[0]
        else:
            where_clause = " AND ".join(where)
        r = self._db_read("SELECT subj, pred, obj FROM Triples WHERE " + where_clause, *args)
        return cast(List[Tuple[str, str, str]], r)

    def get_all_has_relations(self, memid: str) -> Dict[str, str]:
        all_triples = self.get_triples(subj=memid)
        relation_values = {}
        for _, pred, obj in all_triples:
            if pred.startswith("has_"):
                start = pred.find("has_") + 4
                end = pred.find("-", start)  # FIXME: is this working or a typo??
                name = pred[start:end]
                relation_values[name] = obj
        return relation_values

    def get_objects_by_tags(self, table, *tags) -> List["ObjectNode"]:
        memids = set.intersection(*[set(self.get_memids_by_tag(t)) for t in tags])
        objects = [self.get_object_by_id(memid, table) for memid in memids]
        return [o for o in objects if o is not None]

    def remove_memid_triple(self, memid: str, role="subj"):
        if role == "subj" or role == "both":
            self._db_write("DELETE FROM Triples WHERE subj=?", memid)
        if role == "obj" or role == "both":
            self._db_write("DELETE FROM Triples WHERE obj=?", memid)

    ######################
    ###  BlockObjects  ###
    ######################

    def get_block_object_ids_by_xyz(self, xyz: XYZ) -> List[str]:
        return self.get_object_ids_by_xyz(xyz, "BlockObjects")

    def get_block_object_by_xyz(self, xyz: XYZ) -> Optional["ObjectNode"]:
        memids = self.get_block_object_ids_by_xyz(xyz)
        if len(memids) == 0:
            return None
        return self.get_block_object_by_id(memids[0])

    def get_block_objects_with_tags(self, *tags) -> List["ObjectNode"]:
        tags += ("_block_object",)
        return self.get_objects_by_tags("BlockObjects", *tags)

    def get_block_object_by_id(self, memid: str) -> "ObjectNode":
        return self.get_object_by_id(memid, "BlockObjects")

    def tag_block_object_from_schematic(self, block_object_memid: str, schematic_memid: str):
        self.add_triple(block_object_memid, "_from_schematic", schematic_memid)

    #######################
    ### ComponentObject ###
    #######################

    def get_component_object_by_id(self, memid: str) -> Optional["ObjectNode"]:
        return self.get_object_by_id(memid, "ComponentObjects")

    def get_component_object_ids_by_xyz(self, xyz: XYZ) -> List[str]:
        return self.get_object_ids_by_xyz(xyz, "ComponentObjects")

    def get_component_objects_with_tags(self, *tags) -> List["ObjectNode"]:
        tags += ("_component_object",)
        return self.get_objects_by_tags("ComponentObjects", *tags)

    ####################
    ###  Schematics  ###
    ####################

    def get_schematic_by_id(self, memid: str) -> "SchematicNode":
        return SchematicNode(self, memid)

    def get_schematic_by_name(self, name: str) -> Optional["SchematicNode"]:
        r = self._db_read(
            """
                SELECT Schematics.uuid
                FROM Schematics INNER JOIN Triples as T ON T.subj=Schematics.uuid
                WHERE (T.pred="has_name" OR T.pred="has_tag") AND T.obj=?""",
            name,
        )
        if r:  # if multiple exist, then randomly select one
            return self.get_schematic_by_id(random.choice(r)[0])
        else:
            return None

    def convert_block_object_to_schematic(self, block_object_memid: str) -> "SchematicNode":
        r = self._db_read_one(
            'SELECT subj FROM Triples WHERE pred="_source_block_object" AND obj=?',
            block_object_memid,
        )
        if r:
            # previously converted; return old schematic
            return self.get_schematic_by_id(r[0])

        else:
            # get up to date BlockObject
            block_object = self.get_block_object_by_id(block_object_memid)

            # create schematic
            memid = SchematicNode.create(self, list(block_object.blocks.items()))

            # add triple linking the object to the schematic
            self.add_triple(memid, "_source_block_object", block_object.memid)

            return self.get_schematic_by_id(memid)

    def _load_schematics(self, load_minecraft_specs=True):
        if load_minecraft_specs:
            for name, npys in minecraft_specs.get_schematics().items():
                for npy in npys:
                    memid = SchematicNode.create(self, npy_to_blocks_list(npy))
                    self.add_triple(memid, "has_name", name)

        # load single blocks as schematics
        bid_to_name = minecraft_specs.get_block_data()["bid_to_name"]
        for (d, m), name in bid_to_name.items():
            if d >= 256:
                continue
            memid = SchematicNode.create(self, [((0, 0, 0), (d, m))])
            self.add_triple(memid, "has_name", name)
            if "block" in name:
                self.add_triple(memid, "has_name", name.strip("block").strip())

    ##############
    ###  Mobs  ###
    ##############

    def get_mobs(
        self,
        spatial_range: Tuple[int, int, int, int, int, int] = None,
        spawntime: Tuple[float, float] = None,
        mobtype: str = None,
    ) -> List["MobNode"]:
        """Find mobs matching the given filters

        Args:
          spatial_range = [xmin, xmax, ymin, ymax, zmin, zmax]
          spawntime = [time_min, time_max] .  can be negative to ignore
          mobtype = string
        """
        query = "SELECT uuid FROM Mobs WHERE "
        args: List = []
        if spatial_range is not None:
            args.extend(spatial_range)
            query += " x>? AND x<? AND y>? AND y<? AND z>? AND z<?"
            if spawntime is not None or mobtype is not None:
                query += " AND "
        if spawntime is not None:
            if spawntime[0] > 0:
                args.append(spawntime[0])
                query += " spawn>? "
                if spawntime[1] > 0 or mobtype is not None:
                    query += " AND "
            if spawntime[1] > 0:
                args.append(spawntime[1])
                query += " spawn<? "
                if mobtype is not None:
                    query += " AND "
        if mobtype is not None:
            args.append(mobtype)
            query += " mobtype=?"
        memids = [m[0] for m in self._db_read(query, *args)]
        return [self.get_mob_by_id(memid) for memid in memids]

    def get_mobs_tagged(self, *tags) -> List["MobNode"]:
        tags += ("_mob",)
        memids = set.intersection(*[set(self.get_memids_by_tag(t)) for t in tags])
        return [self.get_mob_by_id(memid) for memid in memids]

    def get_mob_by_id(self, memid) -> "MobNode":
        return MobNode(self, memid)

    def get_mob_by_eid(self, eid) -> Optional["MobNode"]:
        r = self._db_read_one("SELECT uuid FROM Mobs WHERE eid=?", eid)
        if r:
            return MobNode(self, r[0])
        else:
            return None

    def set_mob_position(self, mob) -> "MobNode":
        r = self._db_read_one("SELECT uuid FROM Mobs WHERE eid=?", mob.entityId)
        if r:
            self._db_write(
                "UPDATE Mobs SET x=?, y=?, z=? WHERE eid=?",
                mob.pos.x,
                mob.pos.y,
                mob.pos.z,
                mob.entityId,
            )
            memid, = r
        else:
            memid = MobNode.create(self, mob)
        return self.get_mob_by_id(memid)

    ###############
    ###  Chats  ###
    ###############

    def add_chat(self, speaker_memid: str, chat: str) -> str:
        return ChatNode.create(self, speaker_memid, chat)

    def get_chat_by_id(self, memid: str) -> "ChatNode":
        return ChatNode(self, memid)

    def get_recent_chats(self, n=1) -> List["ChatNode"]:
        """Return a list of at most n chats"""
        r = self._db_read("SELECT uuid FROM Chats ORDER BY time DESC LIMIT ?", n)
        return [ChatNode(self, m) for m, in reversed(r)]

    def get_most_recent_incoming_chat(self, after=-1) -> Optional["ChatNode"]:
        r = self._db_read_one(
            """
            SELECT uuid
            FROM Chats
            WHERE speaker != ? AND time > ?
            ORDER BY time DESC
            LIMIT 1
            """,
            self.self_memid,
            after,
        )
        if r:
            return ChatNode(self, r[0])
        else:
            return None

    ###################
    ###  Locations  ###
    ###################

    def add_location(self, xyz: XYZ) -> str:
        return LocationNode.create(self, xyz)

    def get_location_by_id(self, memid: str) -> "LocationNode":
        return LocationNode(self, memid)

    ###############
    ###  Dances  ##
    ###############

    def add_dance(self, dance_fn):
        # a dance is movement determined as a sequence of steps, rather than by its destination
        memid = uuid.uuid4().hex
        self._db_write("INSERT INTO Dances(uuid) VALUES (?)", memid)
        self.dances[memid] = dance_fn
        return memid

    ###############
    ###  Tasks  ###
    ###############

    def task_stack_push(
        self, task: Task, parent_memid: str = None, chat_effect: bool = False
    ) -> "TaskNode":

        memid = TaskNode.create(self, task)

        # Relations
        if parent_memid:
            self.add_triple(memid, "_has_parent_task", parent_memid)
        if chat_effect:
            chat = self.get_most_recent_incoming_chat()
            assert chat is not None, "chat_effect=True with no incoming chats"
            self.add_triple(chat.memid, "chat_effect_", memid)

        # Return newly created object
        return TaskNode(self, memid)

    def task_stack_update_task(self, memid: str, task: Task):
        self._db_write("UPDATE Tasks SET pickled=? WHERE uuid=?", self.safe_pickle(task), memid)

    def task_stack_peek(self) -> Optional["TaskNode"]:
        r = self._db_read_one(
            """
            SELECT uuid
            FROM Tasks
            WHERE finished_at < 0 AND paused = 0
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        if r:
            return TaskNode(self, r[0])
        else:
            return None

    def task_stack_pop(self) -> Optional["TaskNode"]:
        """Return the 'TaskNode' of the stack head and mark finished"""
        mem = self.task_stack_peek()
        if mem is None:
            raise ValueError("Called task_stack_pop with empty stack")
        self._db_write("UPDATE Tasks SET finished_at=? WHERE uuid=?", time.time(), mem.memid)
        return mem

    def task_stack_pause(self) -> bool:
        """Pause the stack and return True iff anything was stopped"""
        return self._db_write("UPDATE Tasks SET paused=1 WHERE finished_at < 0") > 0

    def task_stack_clear(self):
        self._db_write("DELETE FROM Tasks WHERE finished_at < 0")

    def task_stack_resume(self) -> bool:
        """Resume stopped tasks. Return True if there was something to resume."""
        return self._db_write("UPDATE Tasks SET paused=0") > 0

    def task_stack_find_lowest_instance(
        self, cls_names: Union[str, Sequence[str]]
    ) -> Optional["TaskNode"]:
        """Find and return the lowest item in the stack of the given class(es)"""
        names = [cls_names] if type(cls_names) == str else cls_names
        memid, = self._db_read_one(
            "SELECT uuid FROM Tasks WHERE {} ORDER BY created_at LIMIT 1".format(
                " OR ".join(["action_name=?" for _ in names])
            ),
            *names,
        )

        if memid is not None:
            return TaskNode(self, memid)
        else:
            return None

    def task_stack_get_all(self) -> List["TaskNode"]:
        r = self._db_read(
            """
            SELECT uuid
            FROM Tasks
            WHERE paused=0 AND finished_at<0
            ORDER BY created_at
            """
        )
        return [TaskNode(self, memid) for memid, in r]

    def get_last_finished_root_task(
        self, action_name: str = None, recency: int = 300
    ) -> "TaskNode":
        q = """
        SELECT uuid
        FROM Tasks
        WHERE finished_at > ? {}
        ORDER BY created_at DESC
        """.format(
            " AND action_name=?" if action_name else ""
        )
        args: List = [time.time() - recency]
        if action_name:
            args.append(action_name)
        memids = [r[0] for r in self._db_read(q, *args)]
        for memid in memids:
            if self._db_read_one(
                "SELECT uuid FROM Triples WHERE pred='_has_parent_task' AND subj=?", memid
            ):
                # not a root task
                continue

            return TaskNode(self, memid)
        raise ValueError("Called get_last_finished_root_task with no finished root tasks")

    def get_task_by_id(self, memid: str) -> "TaskNode":
        return TaskNode(self, memid)

    #################
    ###  Players  ###
    #################

    # TODO remove players that leave
    def update_other_players(self, player_list: List):
        for p in player_list:
            mem = self.get_player_by_name(p.name)
            if mem is None:
                memid = PlayerNode.create(self, p)
            else:
                memid = mem.memid
            self.other_players[memid] = p

    def get_player_struct_by_name(self, name):
        """Return a Player struct (not wrapped as a MemoryNode) or None"""
        mem = self.get_player_by_name(name)
        if mem:
            return self.other_players[mem.memid]
        else:
            return None

    def get_player_by_name(self, name) -> Optional["PlayerNode"]:
        r = self._db_read_one("SELECT uuid FROM Players WHERE name=?", name)
        if r:
            return PlayerNode(self, r[0])
        else:
            return None

    def get_player_by_id(self, memid) -> "PlayerNode":
        return PlayerNode(self, memid)

    #########################
    ###  Database Access  ###
    #########################

    def _db_read(self, query, *args) -> List[Tuple]:
        args = tuple(a.item() if isinstance(a, np.number) else a for a in args)
        try:
            c = self.db.cursor()
            c.execute(query, args)
            r = c.fetchall()
            c.close()
            return r
        except:
            logging.error("Bad read: {} : {}".format(query, args))
            raise

    def _db_read_one(self, query, *args) -> Tuple:
        args = tuple(a.item() if isinstance(a, np.number) else a for a in args)
        try:
            c = self.db.cursor()
            c.execute(query, args)
            r = c.fetchone()
            c.close()
            return r
        except:
            logging.error("Bad read: {} : {}".format(query, args))
            raise

    def _db_write(self, query: str, *args) -> int:
        """Return the number of rows affected"""
        args = tuple(a.item() if isinstance(a, np.number) else a for a in args)
        try:
            c = self.db.cursor()
            c.execute(query, args)
            self.db.commit()
            c.close()
            self._write_to_db_log(query, *args)
            return c.rowcount
        except:
            logging.error("Bad write: {} : {}".format(query, args))
            raise

    def _db_script(self, script: str):
        c = self.db.cursor()
        c.executescript(script)
        self.db.commit()
        c.close()
        self._write_to_db_log(script, no_format=True)

    ####################
    ###  DB LOGGING  ###
    ####################

    def get_db_log_idx(self):
        return self._db_log_idx

    def _write_to_db_log(self, s: str, *args, no_format=False):
        if not getattr(self, "_db_log_file", None):
            return

        # sub args in for ?
        split = s.split("?")
        final = b""
        for sub, arg in zip_longest(split, args, fillvalue=""):
            final += str(sub).encode("utf-8")
            if type(arg) == str and arg != "":
                # put quotes around string args
                final += '"{}"'.format(arg).encode("utf-8")
            else:
                final += str(arg).encode("utf-8")

        # remove newlines, add semicolon
        if not no_format:
            final = final.strip().replace(b"\n", b" ") + b";\n"

        # write to file
        self._db_log_file.write(final)
        self._db_log_file.flush()
        self._db_log_idx += 1

    ######################
    ###  MISC HELPERS  ###
    ######################

    def dump(self, sql_file, dict_memory_file=None):
        sql_file.write("\n".join(self.db.iterdump()))
        if dict_memory_file is not None:
            import io
            import pickle

            assert type(dict_memory_file) == io.BufferedWriter
            dict_memory = {"task_db": self.task_db}
            pickle.dump(dict_memory, dict_memory_file)

    def _is_placed_block_interesting(self, xyz: XYZ, bid: int) -> Tuple[bool, bool]:
        """Return two values:
        - bool: is the placed block interesting?
        - bool: is it interesting because it was placed by a player?
        """
        if xyz in self.pending_agent_placed_blocks:
            self.pending_agent_placed_blocks.remove(xyz)
            return True, True
        for player in self.other_players.values():
            if util.euclid_dist(util.pos_to_np(player.pos), xyz) < 5 and player.mainHand.id == bid:
                return True, True
        if bid not in BORING_BLOCKS:
            return True, False
        return False, False

    def safe_pickle(self, obj):
        # little bit scary...
        if not hasattr(obj, "pickled_attrs_id"):
            if hasattr(obj, "memid"):
                obj.pickled_attrs_id = obj.memid
            else:
                try:
                    obj.pickled_attrs_id = uuid.uuid4().hex
                except:
                    pass
        for attr in ["memory", "agent_memory", "new_tasks_fn", "stop_condition", "movement"]:
            if hasattr(obj, attr):
                if self._safe_pickle_saved_attrs.get(obj.pickled_attrs_id) is None:
                    self._safe_pickle_saved_attrs[obj.pickled_attrs_id] = {}
                val = getattr(obj, attr)
                delattr(obj, attr)
                setattr(obj, "__had_attr_" + attr, True)
                self._safe_pickle_saved_attrs[obj.pickled_attrs_id][attr] = val
        return pickle.dumps(obj)

    def safe_unpickle(self, bs):
        obj = pickle.loads(bs)
        if hasattr(obj, "pickled_attrs_id"):
            for attr in ["memory", "agent_memory", "new_tasks_fn", "stop_condition", "movement"]:
                if hasattr(obj, "__had_attr_" + attr):
                    delattr(obj, "__had_attr_" + attr)
                    setattr(obj, attr, self._safe_pickle_saved_attrs[obj.pickled_attrs_id][attr])
        return obj


##################
###  MEMORIES  ###
##################


class MemoryNode:
    PROPERTIES_BLACKLIST = ["agent_memory", "forgetme"]
    TABLE: Optional[str] = None

    @classmethod
    def new(cls, agent_memory) -> str:
        memid = uuid.uuid4().hex
        agent_memory._db_write("INSERT INTO Memories VALUES (?,?,?)", memid, cls.TABLE, 0)
        return memid

    def __init__(self, agent_memory: AgentMemory, memid: str):
        self.agent_memory = agent_memory
        self.memid = memid

    def get_tags(self) -> List[str]:
        return self.agent_memory.get_tags_by_memid(self.memid)

    def get_all_has_relations(self) -> Dict[str, str]:
        return self.agent_memory.get_all_has_relations(self.memid)

    def get_properties(self) -> Dict[str, str]:
        blacklist = self.PROPERTIES_BLACKLIST + self._more_properties_blacklist()
        return {k: v for k, v in self.__dict__.items() if k not in blacklist}

    def update_recently_used(self) -> None:
        self.agent_memory._db_write(
            "UPDATE Memories SET workspace_updated_time=? WHERE uuid=?", self.memid, time.time()
        )

    def _more_properties_blacklist(self) -> List[str]:
        """Override in subclasses to add additional keys to the properties blacklist"""
        return []


class ReferenceObjectNode(MemoryNode):
    def get_pos(self) -> XYZ:
        raise NotImplementedError("must be implemented in subclass")


class ObjectNode(ReferenceObjectNode):
    def __init__(self, agent_memory: AgentMemory, memid: str, last_placed: XYZ):
        super().__init__(agent_memory, memid)
        self.last_placed = last_placed
        r = self.agent_memory._db_read(
            "SELECT x, y, z, bid, meta FROM {} WHERE uuid=?".format(self.TABLE), self.memid
        )
        self.blocks = {(x, y, z): (b, m) for (x, y, z, b, m) in r}

    def get_pos(self) -> XYZ:
        return cast(XYZ, tuple(int(x) for x in np.mean(list(self.blocks.keys()), axis=0)))


class BlockObjectNode(ObjectNode):
    TABLE = "BlockObjects"

    @classmethod
    def create(cls, memory, blocks: Sequence[Block]) -> str:
        # check if block object already exists in memory
        for xyz, _ in blocks:
            memids = memory.get_block_object_ids_by_xyz(xyz)
            if memids:
                return memids[0]

        memid = cls.new(memory)
        for block in blocks:
            memory._upsert_block(block, memid, "BlockObjects")
        memory.tag(memid, "_block_object")
        memory.tag(memid, "_physical_object")
        logging.info(
            "Added block object {} with {} blocks, {}".format(
                memid, len(blocks), Counter([idm for _, idm in blocks])
            )
        )
        return memid

    def __repr__(self):
        return "<BlockObject Node @ {}>".format(list(self.blocks.keys())[0])


class ComponentObjectNode(ObjectNode):
    TABLE = "ComponentObjects"

    @classmethod
    def create(cls, memory: AgentMemory, blocks: List[Block], labels: List[str]) -> str:
        memid = cls.new(memory)
        for block in blocks:
            memory._upsert_block(block, memid, cls.TABLE)
        memory.tag(memid, "_component_object")
        memory.tag(memid, "_physical_object")
        for l in labels:
            memory.tag(memid, l)
        return memid


class MobNode(ReferenceObjectNode):
    TABLE = "Mobs"

    def __init__(self, agent_memory: AgentMemory, memid: str):
        super().__init__(agent_memory, memid)
        eid, x, y, z = self.agent_memory._db_read_one(
            "SELECT eid, x, y, z FROM Mobs WHERE uuid=?", self.memid
        )
        self.eid = eid
        self.pos = (x, y, z)

    @classmethod
    def create(cls, memory: AgentMemory, mob) -> str:
        memid = cls.new(memory)
        mobtype = MOBS_BY_ID[mob.mobType]
        memory._db_write(
            "INSERT INTO Mobs(uuid, eid, x, y, z, mobtype, spawn) VALUES (?, ?, ?, ?, ?, ?, ?)",
            memid,
            mob.entityId,
            mob.pos.x,
            mob.pos.y,
            mob.pos.z,
            mobtype,
            time.time(),
        )
        memory.tag(memid, "_mob")
        memory.tag(memid, mobtype)
        return memid

    def get_pos(self) -> XYZ:
        return self.pos


class PlayerNode(ReferenceObjectNode):
    TABLE = "Players"

    def __init__(self, agent_memory: AgentMemory, memid: str):
        super().__init__(agent_memory, memid)
        # TODO: store in sqlite
        player_struct = self.agent_memory.other_players[self.memid]
        self.pos = player_struct.pos
        self.look = player_struct.look
        self.eid = player_struct.entityId
        self.name = player_struct.name

    @classmethod
    def create(cls, memory: AgentMemory, player_struct) -> str:
        memid = cls.new(memory)
        memory._db_write("INSERT INTO Players(uuid, name) VALUES (?,?)", memid, player_struct.name)
        return memid

    def get_pos(self) -> XYZ:
        return self.pos


class SchematicNode(MemoryNode):
    TABLE = "Schematics"

    def __init__(self, agent_memory: AgentMemory, memid: str):
        super().__init__(agent_memory, memid)
        r = self.agent_memory._db_read(
            "SELECT x, y, z, bid, meta FROM Schematics WHERE uuid=?", self.memid
        )
        self.blocks = {(x, y, z): (b, m) for (x, y, z, b, m) in r}

    @classmethod
    def create(cls, memory: AgentMemory, blocks: Sequence[Block]) -> str:
        memid = cls.new(memory)
        for ((x, y, z), (b, m)) in blocks:
            memory._db_write(
                """
                    INSERT INTO Schematics(uuid, x, y, z, bid, meta)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                memid,
                x,
                y,
                z,
                b,
                m,
            )
        return memid


class LocationNode(MemoryNode):
    TABLE = "Locations"

    def __init__(self, agent_memory: AgentMemory, memid: str):
        super().__init__(agent_memory, memid)
        x, y, z = self.agent_memory._db_read_one(
            "SELECT x, y, z FROM Locations WHERE uuid=?", self.memid
        )
        self.location = (x, y, z)

    @classmethod
    def create(cls, memory: AgentMemory, xyz: XYZ) -> str:
        memid = cls.new(memory)
        memory._db_write(
            "INSERT INTO Locations(uuid, x, y, z) VALUES (?, ?, ?, ?)",
            memid,
            xyz[0],
            xyz[1],
            xyz[2],
        )
        return memid


class ChatNode(MemoryNode):
    TABLE = "Chats"

    def __init__(self, agent_memory: AgentMemory, memid: str):
        super().__init__(agent_memory, memid)
        speaker, chat_text, time = self.agent_memory._db_read_one(
            "SELECT speaker, chat, time FROM Chats WHERE uuid=?", self.memid
        )
        self.speaker_id = speaker
        self.chat_text = chat_text
        self.time = time

    @classmethod
    def create(cls, memory: AgentMemory, speaker: str, chat: str) -> str:
        memid = cls.new(memory)
        memory._db_write(
            "INSERT INTO Chats(uuid, speaker, chat, time) VALUES (?, ?, ?, ?)",
            memid,
            speaker,
            chat,
            time.time(),
        )
        return memid


class TaskNode(MemoryNode):
    TABLE = "Tasks"

    def __init__(self, agent_memory: AgentMemory, memid: str):
        super().__init__(agent_memory, memid)
        pickled, created_at, finished_at, action_name = self.agent_memory._db_read_one(
            "SELECT pickled, created_at, finished_at, action_name FROM Tasks WHERE uuid=?", memid
        )
        self.task = self.agent_memory.safe_unpickle(pickled)
        self.created_at = created_at
        self.finished_at = finished_at
        self.action_name = action_name

    @classmethod
    def create(cls, memory: AgentMemory, task: Task) -> str:
        memid = cls.new(memory)
        task.memid = memid  # FIXME: this shouldn't be necessary, merge Task and TaskNode?
        memory._db_write(
            "INSERT INTO Tasks (uuid, action_name, pickled, created_at) VALUES (?,?,?,?)",
            memid,
            task.__class__.__name__,
            memory.safe_pickle(task),
            time.time(),
        )
        return memid

    def get_chat(self) -> Optional[ChatNode]:
        """Return the memory of the chat that caused this task's creation, or None"""
        triples = self.agent_memory.get_triples(pred="chat_effect_", obj=self.memid)
        if triples:
            chat_id, _, _ = triples[0]
            return ChatNode(self.agent_memory, chat_id)
        else:
            return None

    def get_parent_task(self) -> Optional["TaskNode"]:
        """Return the 'TaskNode' of the parent task, or None"""
        triples = self.agent_memory.get_triples(subj=self.memid, pred="_has_parent_task")
        if len(triples) == 0:
            return None
        elif len(triples) == 1:
            _, _, parent_memid = triples[0]
            return TaskNode(self.agent_memory, parent_memid)
        else:
            raise AssertionError("Task {} has multiple parents: {}".format(self.memid, triples))

    def get_root_task(self) -> Optional["TaskNode"]:
        mem = self
        parent = self.get_parent_task()
        while parent is not None:
            mem = parent
            parent = mem.get_parent_task()
        return mem

    def get_child_tasks(self) -> List["TaskNode"]:
        """Return tasks that were spawned beause of this task"""
        r = self.agent_memory.get_triples(pred="_has_parent_task", obj=self.memid)
        memids = [m for m, _, _ in r]
        return [TaskNode(self.agent_memory, m) for m in memids]

    def all_descendent_tasks(self, include_root=False) -> List["TaskNode"]:
        """Return a list of 'TaskNode' objects whose _has_parent_task root is this task

        If include_root is True, include this node in the list.

        Tasks are returned in the order they were finished.
        """
        descendents = []
        q = [self]
        while q:
            task = q.pop()
            children = task.get_child_tasks()
            descendents.extend(children)
            q.extend(children)
        if include_root:
            descendents.append(self)
        return sorted(descendents, key=lambda t: t.finished_at)

    def __repr__(self):
        return "<TaskNode: {}>".format(self.task)


class RewardNode(MemoryNode):
    TABLE = "Rewards"

    def __init__(self, memory: AgentMemory, memid: str):
        _, value, timestamp = memory._db_read_one("SELECT * FROM Rewards WHERE uuid=?", memid)
        self.value = value
        self.time = timestamp

    @classmethod
    def create(cls, memory: AgentMemory, reward_value: str) -> str:
        memid = cls.new(memory)
        memory._db_write(
            "INSERT INTO Rewards(uuid, value, time) VALUES (?,?,?)",
            memid,
            reward_value,
            time.time(),
        )
        return memid
