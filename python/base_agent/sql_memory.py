"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

###TODO put dances back
import gzip
import logging
import numpy as np
import os
import pickle
import sqlite3
import uuid
from itertools import zip_longest
from typing import cast, Optional, List, Tuple, Dict, Sequence, Union
from base_agent.util import XYZ, Time
from base_agent.task import Task

from base_agent.memory_nodes import (  # noqa
    TaskNode,
    PlayerNode,
    MemoryNode,
    ChatNode,
    TimeNode,
    LocationNode,
    ReferenceObjectNode,
    NODELIST,
)


SCHEMAS = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]

# TODO when a memory is removed, its last state should be snapshotted to prevent tag weirdness


class AgentMemory:
    def __init__(
        self, db_file=":memory:", schema_paths=SCHEMAS, db_log_path=None, nodelist=NODELIST
    ):
        if db_log_path:
            self._db_log_file = gzip.open(db_log_path + ".gz", "w")
            self._db_log_idx = 0
        self.sql_queries = []
        self.db = sqlite3.connect(db_file)
        self.task_db = {}
        self.time = Time(mode="clock")
        self._safe_pickle_saved_attrs = {}

        for schema_path in schema_paths:
            with open(schema_path, "r") as f:
                self._db_script(f.read())

        self.all_tables = [
            c[0] for c in self._db_read("SELECT name FROM sqlite_master WHERE type='table';")
        ]
        self.nodes = {}
        for node in nodelist:
            self.nodes[node.TABLE] = node

        # create a "self" memory to reference in Triples
        self.self_memid = "0" * len(uuid.uuid4().hex)
        self._db_write(
            "INSERT INTO Memories VALUES (?,?,?,?,?,?)",
            self.self_memid,
            "Memories",
            0,
            0,
            -1,
            False,
        )
        self.tag(self.self_memid, "_agent")
        self.tag(self.self_memid, "_self")

    def __del__(self):
        if getattr(self, "_db_log_file", None):
            self._db_log_file.close()

    def get_time(self):
        return self.time.get_time()

    def add_tick(self, ticks=1):
        self.time.add_tick(ticks)

    # TODO list of all "updatable" mems, do a mem.update() ?
    def update(self, agent):
        pass

    ########################
    ### Workspace memory ###
    ########################

    def set_memory_updated_time(self, memid):
        self._db_write("UPDATE Memories SET updated_time=? WHERE uuid=?", self.get_time(), memid)

    def set_memory_attended_time(self, memid):
        self._db_write("UPDATE Memories SET attended_time=? WHERE uuid=?", self.get_time(), memid)

    def update_recent_entities(self, mems=[]):
        logging.info("update_recent_entities {}".format(mems))
        for mem in mems:
            mem.update_recently_attended()

    # for now, no archives in recent entities
    def get_recent_entities(self, memtype, time_window=12000) -> List["MemoryNode"]:
        r = self._db_read(
            """SELECT uuid
            FROM Memories
            WHERE tabl=? AND attended_time >= ? and is_snapshot=0
            ORDER BY attended_time DESC""",
            memtype,
            self.get_time() - time_window,
        )
        return [self.get_mem_by_id(memid, memtype) for memid, in r]

    ###############
    ### General ###
    ###############

    def get_memid_table(self, memid: str) -> str:
        (r,) = self._db_read_one("SELECT tabl FROM Memories WHERE uuid=?", memid)
        return r

    def get_mem_by_id(self, memid: str, table: str = None) -> "MemoryNode":
        if table is None:
            table = self.get_memid_table(memid)

        if table is None:
            return

        return self.nodes.get(table, MemoryNode)(self, memid)

    # does not search archived mems for now
    def get_all_tagged_mems(self, tag: str) -> List["MemoryNode"]:
        memids = self.get_memids_by_tag(tag)
        return [self.get_mem_by_id(memid) for memid in memids]

    def check_memid_exists(self, memid: str, table: str) -> bool:
        return bool(self._db_read_one("SELECT * FROM {} WHERE uuid=?".format(table), memid))

    # TODO forget should be a method of the memory object
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

    #################
    ###  Triples  ###
    #################

    def add_triple(self, subj: str, pred: str, obj: str, confidence=1.0):
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

    # does not search archived mems for now
    def get_memids_by_tag(self, tag: str) -> List[str]:
        r = self._db_read(
            'SELECT DISTINCT(Memories.uuid) FROM Memories INNER JOIN Triples as T ON T.subj=Memories.uuid WHERE T.pred="has_tag" AND T.obj=? AND Memories.is_snapshot=0',
            tag,
        )
        #        r = self._db_read('SELECT DISTINCT(subj) FROM Triples WHERE pred="has_tag" AND obj=?', tag)
        return [x for (x,) in r]

    # TODO rename me get_tags_by_subj
    def get_tags_by_memid(self, subj_memid: str) -> List[str]:
        r = self._db_read(
            'SELECT DISTINCT(obj) FROM Triples WHERE pred="has_tag" AND subj=?', subj_memid
        )
        return [x for (x,) in r]

    # does not search archived mems for now
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
        r = self._db_read(
            "SELECT subj, pred, obj FROM Triples INNER JOIN Memories as M ON Triples.subj=M.uuid WHERE M.is_snapshot=0 AND "
            + where_clause,
            *args,
        )
        #        r = self._db_read("SELECT subj, pred, obj FROM Triples WHERE " + where_clause, *args)
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

    def remove_memid_triple(self, memid: str, role="subj"):
        if role == "subj" or role == "both":
            self._db_write("DELETE FROM Triples WHERE subj=?", memid)
        if role == "obj" or role == "both":
            self._db_write("DELETE FROM Triples WHERE obj=?", memid)

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
            WHERE speaker != ? AND time >= ?
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

    #################
    ###  Players  ###
    #################

    def get_player_by_eid(self, eid) -> Optional["PlayerNode"]:
        r = self._db_read_one(
            "SELECT Players.uuid FROM Players INNER JOIN Memories as M ON Players.uuid=M.uuid WHERE M.is_snapshot=0 AND eid=?",
            eid,
        )
        #        r = self._db_read_one("SELECT uuid FROM Players WHERE name=?", name)
        if r:
            return PlayerNode(self, r[0])
        else:
            return None

    def get_player_by_name(self, name) -> Optional["PlayerNode"]:
        r = self._db_read_one(
            "SELECT Players.uuid FROM Players INNER JOIN Memories as M ON Players.uuid=M.uuid WHERE M.is_snapshot=0 AND name=?",
            name,
        )
        #        r = self._db_read_one("SELECT uuid FROM Players WHERE name=?", name)
        if r:
            return PlayerNode(self, r[0])
        else:
            return None

    def get_players_tagged(self, *tags) -> List["PlayerNode"]:
        tags += ("_player",)
        memids = set.intersection(*[set(self.get_memids_by_tag(t)) for t in tags])
        return [self.get_player_by_id(memid) for memid in memids]

    def get_player_by_id(self, memid) -> "PlayerNode":
        return PlayerNode(self, memid)

    ###################
    ###  Locations  ###
    ###################

    def add_location(self, xyz: XYZ) -> str:
        return LocationNode.create(self, xyz)

    def get_location_by_id(self, memid: str) -> "LocationNode":
        return LocationNode(self, memid)

    ###############
    ###  Times  ###
    ###############

    def add_time(self, t: int) -> str:
        return TimeNode.create(self, t)

    def get_time_by_id(self, memid: str) -> "TimeNode":
        return TimeNode(self, memid)

    #    ###############
    #    ###  Sets   ###
    #    ###############
    #
    #    def add_set(self, memid_list):
    #        set_memid = SetNode.create(self)
    #        self.add_objs_to_set(set_memid, memid_list)
    #        return SetNode(self, set_memid)
    #
    #    def add_objs_to_set(self, set_memid, memid_list):
    #        for mid in memid_list:
    #            self.add_triple(mid, "set_member_", set_memid)

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
        self._db_write("UPDATE Tasks SET finished_at=? WHERE uuid=?", self.get_time(), mem.memid)
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
        (memid,) = self._db_read_one(
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

    def get_last_finished_root_task(self, action_name: str = None, recency: int = None):
        q = """
        SELECT uuid
        FROM Tasks
        WHERE finished_at >= ? {}
        ORDER BY created_at DESC
        """.format(
            " AND action_name=?" if action_name else ""
        )
        if recency is None:
            recency = self.time.round_time(300)
        args: List = [self.get_time() - recency]
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

    #        raise ValueError("Called get_last_finished_root_task with no finished root tasks")

    def get_task_by_id(self, memid: str) -> "TaskNode":
        return TaskNode(self, memid)

    #################
    ###   Time    ###
    #################

    def hurry_up(self):
        if self.task_stack_peek() is None:
            return  # send chat?
        task_mem = self.task_stack_peek()
        task_mem.task.hurry_up()
        self.task_stack_update_task(task_mem.memid, task_mem.task)

    def slow_down(self):
        if self.task_stack_peek() is None:
            return  # send chat?
        task_mem = self.task_stack_peek()
        task_mem.task.slow_down()
        self.task_stack_update_task(task_mem.memid, task_mem.task)

    #########################
    ###  Database Access  ###
    #########################

    def _db_read(self, query, *args) -> List[Tuple]:
        args = tuple(a.item() if isinstance(a, np.number) else a for a in args)
        try:
            c = self.db.cursor()
            c.execute(query, args)
            query = query.replace("?", "{}").format(*args)
            if query not in self.sql_queries:
                self.sql_queries.append(query)
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
            query = query.replace("?", "{}").format(*args)
            if query not in self.sql_queries:
                self.sql_queries.append(query)
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
            query = query.replace("?", "{}").format(*args)
            if query not in self.sql_queries:
                self.sql_queries.append(query)
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
