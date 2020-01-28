import numpy as np
import uuid
import logging
from collections import Counter
from typing import cast, Optional, List, Dict, Sequence
from util import XYZ, POINT_AT_TARGET, IDM, Block, get_bounds
from entities import MOBS_BY_ID
from task import Task


class MemoryNode:
    PROPERTIES_BLACKLIST = ["agent_memory", "forgetme"]
    TABLE: Optional[str] = None

    @classmethod
    def new(cls, agent_memory) -> str:
        memid = uuid.uuid4().hex
        agent_memory._db_write("INSERT INTO Memories VALUES (?,?,?)", memid, cls.TABLE, 0)
        return memid

    def __init__(self, agent_memory, memid: str):
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
        self.agent_memory.set_memory_updated_time(self.memid)

    def _more_properties_blacklist(self) -> List[str]:
        """Override in subclasses to add additional keys to the properties blacklist"""
        return []


# the table entry just has the memid and a modification time,
# actual set elements are handled as triples
class SetNode(MemoryNode):
    TABLE = "SetMems"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        time = self.agent_memory._db_read_one("SELECT time FROM SetMems WHERE uuid=?", self.memid)
        self.time = time

    @classmethod
    def create(cls, memory) -> str:
        memid = cls.new(memory)
        memory._db_write("INSERT INTO SetMems(uuid, time) VALUES (?, ?)", memid, memory.get_time())
        return memid

    def get_members(self):
        return self.agent_memory.get_triples(pred="set_member_", obj=self.memid)


class ReferenceObjectNode(MemoryNode):
    def get_pos(self) -> XYZ:
        raise NotImplementedError("must be implemented in subclass")

    def get_point_at_target(self) -> POINT_AT_TARGET:
        raise NotImplementedError("must be implemented in subclass")

    def get_bounds(self):
        raise NotImplementedError("must be implemented in subclass")


class ObjectNode(ReferenceObjectNode):
    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        r = self.agent_memory._db_read(
            "SELECT x, y, z, bid, meta FROM {} WHERE uuid=?".format(self.TABLE), self.memid
        )
        self.blocks = {(x, y, z): (b, m) for (x, y, z, b, m) in r}

    def get_pos(self) -> XYZ:
        return cast(XYZ, tuple(int(x) for x in np.mean(list(self.blocks.keys()), axis=0)))

    def get_point_at_target(self) -> POINT_AT_TARGET:
        point_min = [int(x) for x in np.min(list(self.blocks.keys()), axis=0)]
        point_max = [int(x) for x in np.max(list(self.blocks.keys()), axis=0)]
        return cast(POINT_AT_TARGET, point_min + point_max)

    def get_bounds(self):
        return get_bounds(list(self.blocks.keys()))


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
    def create(cls, memory, blocks: List[Block], labels: List[str]) -> str:
        memid = cls.new(memory)
        for block in blocks:
            memory._upsert_block(block, memid, cls.TABLE)
        memory.tag(memid, "_component_object")
        memory.tag(memid, "_physical_object")
        for l in labels:
            memory.tag(memid, l)
        return memid


# note: instance segmentation objects should not be tagged except by the creator
# build an archive if you want to tag permanently
class InstSegNode(ReferenceObjectNode):
    TABLE = "InstSeg"

    @classmethod
    def create(cls, memory, locs, tags=[]) -> str:
        # TODO option to not overwrite
        # check if instance segmentation object already exists in memory
        inst_memids = {}
        for xyz in locs:
            m = memory._db_read("SELECT uuid from InstSeg WHERE x=? AND y=? AND z=?", *xyz)
            if len(m) > 0:
                for memid in m:
                    inst_memids[memid[0]] = True
        for m in inst_memids.keys():
            olocs = memory._db_read("SELECT x, y, z from InstSeg WHERE uuid=?", m)
            # TODO maybe make an archive?
            if len(set(olocs) - set(locs)) == 0:
                memory._db_write("DELETE FROM Memories WHERE uuid=?", m)

        memid = cls.new(memory)
        for loc in locs:
            cmd = "INSERT INTO InstSeg (uuid, x, y, z) VALUES ( ?, ?, ?, ?)"
            memory._db_write(cmd, memid, loc[0], loc[1], loc[2])
        memory.tag(memid, "_inst_seg")
        for tag in tags:
            memory.tag(memid, tag)
        return memid

    def __init__(self, memory, memid: str):
        super().__init__(memory, memid)
        r = memory._db_read("SELECT x, y, z FROM InstSeg WHERE uuid=?", self.memid)
        self.locs = r
        self.blocks = {l: (0, 0) for l in self.locs}
        tags = memory.get_triples(subj=self.memid, pred="has_tag")
        self.tags = []  # noqa: T484
        for tag in tags:
            if tag[2][0] != "_":
                self.tags.append(tag[2])

    def get_pos(self) -> XYZ:
        return cast(XYZ, tuple(int(x) for x in np.mean(self.locs, axis=0)))

    def get_point_at_target(self) -> POINT_AT_TARGET:
        point_min = [int(x) for x in np.min(list(self.blocks.keys()), axis=0)]
        point_max = [int(x) for x in np.max(list(self.blocks.keys()), axis=0)]
        return cast(POINT_AT_TARGET, point_min + point_max)

    def get_bounds(self):
        M = np.max(self.locs, axis=0)
        m = np.min(self.locs, axis=0)
        return m[0], M[0], m[1], M[1], m[2], M[2]

    def __repr__(self):
        return "<InstSeg Node @ {} with tags {} >".format(self.locs, self.tags)


class MobNode(ReferenceObjectNode):
    TABLE = "Mobs"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        eid, x, y, z = self.agent_memory._db_read_one(
            "SELECT eid, x, y, z FROM Mobs WHERE uuid=?", self.memid
        )
        self.eid = eid
        self.pos = (x, y, z)

    @classmethod
    def create(cls, memory, mob, player_placed=False, agent_placed=False) -> str:
        # TODO warn/error if mob already in memory?
        memid = cls.new(memory)
        mobtype = MOBS_BY_ID[mob.mobType]
        memory._db_write(
            "INSERT INTO Mobs(uuid, eid, x, y, z, mobtype, player_placed, agent_placed, spawn) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            memid,
            mob.entityId,
            mob.pos.x,
            mob.pos.y,
            mob.pos.z,
            mobtype,
            player_placed,
            agent_placed,
            memory.get_time(),
        )
        memory.tag(memid, "_mob")
        memory.tag(memid, mobtype)
        return memid

    def get_pos(self) -> XYZ:
        x, y, z = self.agent_memory._db_read_one(
            "SELECT x, y, z FROM Mobs WHERE uuid=?", self.memid
        )
        self.pos = (x, y, z)
        return self.pos

    # TODO: use a smarter way to get point_at_target
    def get_point_at_target(self) -> POINT_AT_TARGET:
        x, y, z = self.agent_memory._db_read_one(
            "SELECT x, y, z FROM Mobs WHERE uuid=?", self.memid
        )
        # use the block above the mob as point_at_target
        return cast(POINT_AT_TARGET, (x, y + 1, z, x, y + 1, z))

    def get_bounds(self):
        x, y, z = self.pos
        return x, x, y, y, z, z


class PlayerNode(ReferenceObjectNode):
    TABLE = "Players"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        # TODO: store in sqlite
        player_struct = self.agent_memory.other_players[self.memid]
        self.pos = player_struct.pos
        self.look = player_struct.look
        self.eid = player_struct.entityId
        self.name = player_struct.name

    @classmethod
    def create(cls, memory, player_struct) -> str:
        memid = cls.new(memory)
        memory._db_write("INSERT INTO Players(uuid, name) VALUES (?,?)", memid, player_struct.name)
        memory.tag(memid, "_player")
        return memid

    def get_pos(self) -> XYZ:
        return self.pos

    # TODO: use a smarter way to get point_at_target
    def get_point_at_target(self) -> POINT_AT_TARGET:
        x, y, z = self.pos
        # use the block above the player as point_at_target
        return cast(POINT_AT_TARGET, (x, y + 1, z, x, y + 1, z))

    def get_bounds(self):
        x, y, z = self.pos
        return x, x, y, y, z, z


class SchematicNode(MemoryNode):
    TABLE = "Schematics"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        r = self.agent_memory._db_read(
            "SELECT x, y, z, bid, meta FROM Schematics WHERE uuid=?", self.memid
        )
        self.blocks = {(x, y, z): (b, m) for (x, y, z, b, m) in r}

    @classmethod
    def create(cls, memory, blocks: Sequence[Block]) -> str:
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


class BlockTypeNode(MemoryNode):
    TABLE = "BlockTypes"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        type_name, b, m = self.agent_memory._db_read(
            "SELECT type_name, bid, meta FROM BlockTypes WHERE uuid=?", self.memid
        )
        self.type_name = type_name
        self.b = b
        self.m = m

    @classmethod
    def create(cls, memory, type_name: str, idm: IDM) -> str:
        memid = cls.new(memory)
        memory._db_write(
            "INSERT INTO BlockTypes(uuid, type_name, bid, meta) VALUES (?, ?, ?, ?)",
            memid,
            type_name,
            idm[0],
            idm[1],
        )
        return memid


# shouldn't this be a reference object?  TODO!!
class LocationNode(MemoryNode):
    TABLE = "Locations"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        x, y, z = self.agent_memory._db_read_one(
            "SELECT x, y, z FROM Locations WHERE uuid=?", self.memid
        )
        self.location = (x, y, z)

    @classmethod
    def create(cls, memory, xyz: XYZ) -> str:
        memid = cls.new(memory)
        memory._db_write(
            "INSERT INTO Locations(uuid, x, y, z) VALUES (?, ?, ?, ?)",
            memid,
            xyz[0],
            xyz[1],
            xyz[2],
        )
        return memid


class TimeNode(MemoryNode):
    TABLE = "Times"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        t = self.agent_memory._db_read_one("SELECT time FROM Times WHERE uuid=?", self.memid)
        self.time = t

    @classmethod
    def create(cls, memory, time: int) -> str:
        memid = cls.new(memory)
        memory._db_write("INSERT INTO Times(uuid, time) VALUES (?, ?)", memid, time)
        return memid


class DanceNode(MemoryNode):
    TABLE = "Dances"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        # TODO put in DB/pickle like tasks?
        self.dance_fn = self.agent_memory[memid]

    @classmethod
    def create(cls, memory, dance_fn, name=None) -> str:
        memid = cls.new(memory)
        memory._db_write("INSERT INTO Dances(uuid) VALUES (?)", memid)
        # TODO put in db via pickle like tasks?
        memory.dances[memid] = dance_fn
        if name is not None:
            memory.add_triple(memid, "has_name", name)
        return memid


class ChatNode(MemoryNode):
    TABLE = "Chats"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        speaker, chat_text, time = self.agent_memory._db_read_one(
            "SELECT speaker, chat, time FROM Chats WHERE uuid=?", self.memid
        )
        self.speaker_id = speaker
        self.chat_text = chat_text
        self.time = time

    @classmethod
    def create(cls, memory, speaker: str, chat: str) -> str:
        memid = cls.new(memory)
        memory._db_write(
            "INSERT INTO Chats(uuid, speaker, chat, time) VALUES (?, ?, ?, ?)",
            memid,
            speaker,
            chat,
            memory.get_time(),
        )
        return memid


class TaskNode(MemoryNode):
    TABLE = "Tasks"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        pickled, created_at, finished_at, action_name = self.agent_memory._db_read_one(
            "SELECT pickled, created_at, finished_at, action_name FROM Tasks WHERE uuid=?", memid
        )
        self.task = self.agent_memory.safe_unpickle(pickled)
        self.created_at = created_at
        self.finished_at = finished_at
        self.action_name = action_name

    @classmethod
    def create(cls, memory, task: Task) -> str:
        memid = cls.new(memory)
        task.memid = memid  # FIXME: this shouldn't be necessary, merge Task and TaskNode?
        memory._db_write(
            "INSERT INTO Tasks (uuid, action_name, pickled, created_at) VALUES (?,?,?,?)",
            memid,
            task.__class__.__name__,
            memory.safe_pickle(task),
            memory.get_time(),
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

    def __init__(self, agent_memory, memid: str):
        _, value, timestamp = agent_memory._db_read_one(
            "SELECT * FROM Rewards WHERE uuid=?", memid
        )
        self.value = value
        self.time = timestamp

    @classmethod
    def create(cls, agent_memory, reward_value: str) -> str:
        memid = cls.new(agent_memory)
        agent_memory._db_write(
            "INSERT INTO Rewards(uuid, value, time) VALUES (?,?,?)",
            memid,
            reward_value,
            agent_memory.get_time(),
        )
        return memid
