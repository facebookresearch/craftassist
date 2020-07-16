import uuid
import ast
from typing import Optional, List, Dict, cast
from .util import XYZ, POINT_AT_TARGET, to_player_struct
from task import Task


class MemoryNode:
    TABLE_COLUMNS = ["uuid"]
    PROPERTIES_BLACKLIST = ["agent_memory", "forgetme"]
    NODE_TYPE: Optional[str] = None

    @classmethod
    def new(cls, agent_memory, snapshot=False) -> str:
        memid = uuid.uuid4().hex
        t = agent_memory.get_time()
        agent_memory._db_write(
            "INSERT INTO Memories VALUES (?,?,?,?,?,?)", memid, cls.NODE_TYPE, t, t, t, snapshot
        )
        return memid

    def __init__(self, agent_memory, memid: str):
        self.agent_memory = agent_memory
        self.memid = memid

    def get_tags(self) -> List[str]:
        return self.agent_memory.get_tags_by_memid(self.memid)

    def get_properties(self) -> Dict[str, str]:
        blacklist = self.PROPERTIES_BLACKLIST + self._more_properties_blacklist()
        return {k: v for k, v in self.__dict__.items() if k not in blacklist}

    def update_recently_attended(self) -> None:
        self.agent_memory.set_memory_attended_time(self.memid)
        self.snapshot(self.agent_memory)

    def _more_properties_blacklist(self) -> List[str]:
        """Override in subclasses to add additional keys to the properties blacklist"""
        return []

    def snapshot(self, agent_memory):
        """Override in subclasses if necessary to properly snapshot."""

        read_cmd = "SELECT "
        for r in self.TABLE_COLUMNS:
            read_cmd += r + ", "
        read_cmd = read_cmd.strip(", ")
        read_cmd += " FROM " + self.TABLE + " WHERE uuid=?"
        data = agent_memory._db_read_one(read_cmd, self.memid)
        if not data:
            raise ("tried to snapshot nonexistent memory")

        archive_memid = self.new(agent_memory, snapshot=True)
        new_data = list(data)
        new_data[0] = archive_memid

        if hasattr(self, "ARCHIVE_TABLE"):
            archive_table = self.ARCHIVE_TABLE
        else:
            archive_table = self.TABLE
        write_cmd = "INSERT INTO " + archive_table + "("
        qs = ""
        for r in self.TABLE_COLUMNS:
            write_cmd += r + ", "
            qs += "?, "
        write_cmd = write_cmd.strip(", ")
        write_cmd += ") VALUES (" + qs.strip(", ") + ")"
        agent_memory._db_write(write_cmd, *new_data)
        link_archive_to_mem(agent_memory, self.memid, archive_memid)


def link_archive_to_mem(agent_memory, memid, archive_memid):
    agent_memory.add_triple(subj=archive_memid, pred_text="_archive_of", obj=memid)
    agent_memory.add_triple(subj=memid, pred_text="_has_archive", obj=archive_memid)


class ProgramNode(MemoryNode):
    """represents logical forms (outputs from the semantic parser)"""

    TABLE_COLUMNS = ["uuid", "logical_form"]
    TABLE = "Programs"
    NODE_TYPE = "Program"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        text = self.agent_memory._db_read_one(
            "SELECT logical_form FROM Programs WHERE uuid=?", self.memid
        )
        self.logical_form = ast.literal_eval(text)

    @classmethod
    def create(cls, memory, logical_form, snapshot=False) -> str:
        memid = cls.new(memory, snapshot=snapshot)
        memory._db_write(
            "INSERT INTO Programs(uuid, logical_form) VALUES (?,?)", memid, format(logical_form)
        )
        return memid


class NamedAbstractionNode(MemoryNode):
    """a abstract concept with a name, to be used in triples"""

    TABLE_COLUMNS = ["uuid", "name"]
    TABLE = "NamedAbstractions"
    NODE_TYPE = "NamedAbstraction"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        name = self.agent_memory._db_read_one(
            "SELECT name FROM NamedAbstractions WHERE uuid=?", self.memid
        )
        self.name = name

    @classmethod
    def create(cls, memory, name, snapshot=False) -> str:
        memid = memory._db_read_one("SELECT uuid FROM NamedAbstractions WHERE name=?", name)
        if memid:
            return memid[0]
        memid = cls.new(memory, snapshot=snapshot)
        memory._db_write("INSERT INTO NamedAbstractions(uuid, name) VALUES (?,?)", memid, name)
        return memid


# the table entry just has the memid and a modification time,
# actual set elements are handled as triples
class SetNode(MemoryNode):
    """ for representing sets of objects, so that it is easier to build complex relations 
    using RDF/triplestore format.  is currently fetal- not used in main codebase yet """

    TABLE_COLUMNS = ["uuid"]
    TABLE = "SetMems"
    NODE_TYPE = "Set"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)

    # FIXME put the member triples
    @classmethod
    def create(cls, memory, snapshot=False) -> str:
        memid = cls.new(memory, snapshot=snapshot)
        memory._db_write("INSERT INTO SetMems(uuid) VALUES (?)", memid, memory.get_time())
        return memid

    def get_members(self):
        return self.agent_memory.get_triples(pred_text="set_member_", obj=self.memid)

    def snapshot(self, agent_memory):
        return SetNode.create(agent_memory, snapshot=True)


class ReferenceObjectNode(MemoryNode):
    """ generic memory node for anything that has a spatial location and can be
    used a spatial reference (e.g. to the left of the x)."""

    TABLE = "ReferenceObjects"
    NODE_TYPE = "ReferenceObject"
    ARCHIVE_TABLE = "ArchivedReferenceObjects"

    def get_pos(self) -> XYZ:
        raise NotImplementedError("must be implemented in subclass")

    def get_point_at_target(self) -> POINT_AT_TARGET:
        raise NotImplementedError("must be implemented in subclass")

    def get_bounds(self):
        raise NotImplementedError("must be implemented in subclass")


class PlayerNode(ReferenceObjectNode):
    """ represents humans and other agents that can affect the world """

    TABLE_COLUMNS = ["uuid", "eid", "name", "x", "y", "z", "pitch", "yaw", "ref_type"]
    NODE_TYPE = "Player"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        eid, name, x, y, z, pitch, yaw = self.agent_memory._db_read_one(
            "SELECT eid, name, x, y, z, pitch, yaw FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        self.eid = eid
        self.name = name
        self.pos = (x, y, z)
        self.pitch = pitch
        self.yaw = yaw

    @classmethod
    def create(cls, memory, player_struct) -> str:
        memid = cls.new(memory)
        memory._db_write(
            "INSERT INTO ReferenceObjects(uuid, eid, name, x, y, z, pitch, yaw, ref_type) VALUES (?,?,?,?,?,?,?,?,?)",
            memid,
            player_struct.entityId,
            player_struct.name,
            player_struct.pos.x,
            player_struct.pos.y,
            player_struct.pos.z,
            player_struct.look.pitch,
            player_struct.look.yaw,
            "player",
        )
        memory.tag(memid, "_player")
        memory.tag(memid, "_physical_object")
        memory.tag(memid, "_animate")
        # this is a hack until memory_filters does "not"
        memory.tag(memid, "_not_location")
        return memid

    @classmethod
    def update(cls, memory, p, memid) -> str:
        cmd = "UPDATE ReferenceObjects SET eid=?, name=?, x=?,  y=?, z=?, pitch=?, yaw=? WHERE "
        cmd = cmd + "uuid=?"
        memory._db_write(
            cmd, p.entityId, p.name, p.pos.x, p.pos.y, p.pos.z, p.look.pitch, p.look.yaw, memid
        )
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

    def get_struct(self):
        return to_player_struct(self.pos, self.yaw, self.pitch, self.eid, self.name)


class SelfNode(PlayerNode):
    """special PLayerNode for representing the agent's self"""

    TABLE_COLUMNS = ["uuid", "eid", "name", "x", "y", "z", "pitch", "yaw", "ref_type"]
    NODE_TYPE = "Self"


# locations should always be archives?
class LocationNode(ReferenceObjectNode):
    """ReferenceObjectNode representing a raw location (a point in space) """

    TABLE_COLUMNS = ["uuid", "x", "y", "z", "ref_type"]
    NODE_TYPE = "Location"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        x, y, z = self.agent_memory._db_read_one(
            "SELECT x, y, z FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        self.location = (x, y, z)
        self.pos = (x, y, z)

    @classmethod
    def create(cls, memory, xyz: XYZ) -> str:
        memid = cls.new(memory)
        memory._db_write(
            "INSERT INTO ReferenceObjects(uuid, x, y, z, ref_type) VALUES (?, ?, ?, ?, ?)",
            memid,
            xyz[0],
            xyz[1],
            xyz[2],
            "location",
        )
        return memid

    def get_bounds(self):
        x, y, z = self.pos
        return x, x, y, y, z, z

    def get_pos(self) -> XYZ:
        return self.pos

    def get_point_at_target(self) -> POINT_AT_TARGET:
        x, y, z = self.pos
        return cast(POINT_AT_TARGET, (x, y, z, x, y, z))


class TimeNode(MemoryNode):
    """represents a temporal 'location' """

    TABLE_COLUMNS = ["uuid", "time"]
    TABLE = "Times"
    NODE_TYPE = "Time"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        t = self.agent_memory._db_read_one("SELECT time FROM Times WHERE uuid=?", self.memid)
        self.time = t

    @classmethod
    def create(cls, memory, time: int) -> str:
        memid = cls.new(memory)
        memory._db_write("INSERT INTO Times(uuid, time) VALUES (?, ?)", memid, time)
        return memid


class ChatNode(MemoryNode):
    """represents a chat/utterance from another agent/human """

    TABLE_COLUMNS = ["uuid", "speaker", "chat", "time"]
    TABLE = "Chats"
    NODE_TYPE = "Time"

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
    """ represents a task object that was placed on the agent's task_stack """

    TABLE_COLUMNS = ["uuid", "action_name", "pickled", "paused", "created_at", "finished_at"]
    TABLE = "Tasks"
    NODE_TYPE = "Task"

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
        triples = self.agent_memory.get_triples(pred_text="chat_effect_", obj=self.memid)
        if triples:
            chat_id, _, _ = triples[0]
            return ChatNode(self.agent_memory, chat_id)
        else:
            return None

    def get_parent_task(self) -> Optional["TaskNode"]:
        """Return the 'TaskNode' of the parent task, or None"""
        triples = self.agent_memory.get_triples(subj=self.memid, pred_text="_has_parent_task")
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
        r = self.agent_memory.get_triples(pred_text="_has_parent_task", obj=self.memid)
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


# list of nodes to register in memory
NODELIST = [
    TaskNode,
    ChatNode,
    LocationNode,
    SetNode,
    TimeNode,
    PlayerNode,
    SelfNode,
    ProgramNode,
    NamedAbstractionNode,
]
