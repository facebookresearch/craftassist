import os
import sys

import numpy as np
import logging
from collections import Counter
from typing import cast, List, Sequence

# FIXME fix util imports
from util import XYZ, POINT_AT_TARGET, IDM, Block, get_bounds
from entities import MOBS_BY_ID

BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(BASE_AGENT_ROOT)

from base_agent.memory_nodes import link_archive_to_mem, ReferenceObjectNode, MemoryNode, NODELIST


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

    def snapshot(self, agent_memory):
        if self.TABLE == "BlockObjects":
            table = "ArchivedBlockObjects"
        else:
            table = self.TABLE
        archive_memid = self.new(agent_memory, snapshot=True)
        for bid, meta in self.blocks.items():
            agent_memory._upsert_block((bid, meta), archive_memid, table)
        link_archive_to_mem(agent_memory, self.memid, archive_memid)
        return archive_memid


class BlockObjectNode(ObjectNode):
    TABLE_ROWS = ["uuid", "x", "y", "z", "bid", "meta", "agent_placed", "player_placed", "updated"]
    TABLE = "BlockObjects"

    @classmethod
    def create(cls, memory, blocks: Sequence[Block]) -> str:
        # check if block object already exists in memory
        for xyz, _ in blocks:
            old_memids = memory.get_block_object_ids_by_xyz(xyz)
            if old_memids:
                return old_memids[0]
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
    TABLE_ROWS = ["uuid", "x", "y", "z", "bid", "meta", "agent_placed", "player_placed", "updated"]
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
    TABLE_ROWS = ["uuid", "x", "y", "z"]
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

    def snapshot(self, agent_memory):
        archive_memid = self.new(agent_memory, snapshot=True)
        for loc in self.locs:
            cmd = "INSERT INTO InstSeg (uuid, x, y, z) VALUES ( ?, ?, ?, ?)"
            agent_memory._db_write(cmd, archive_memid, loc[0], loc[1], loc[2])
        link_archive_to_mem(agent_memory, self.memid, archive_memid)
        return archive_memid

    def __repr__(self):
        return "<InstSeg Node @ {} with tags {} >".format(self.locs, self.tags)


class MobNode(ReferenceObjectNode):
    TABLE_ROWS = [
        "uuid",
        "eid",
        "x",
        "y",
        "z",
        "mobtype",
        "player_placed",
        "agent_placed",
        "spawn",
    ]
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


class SchematicNode(MemoryNode):
    TABLE_ROWS = ["uuid", "x", "y", "z", "bid", "meta"]
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
    TABLE_ROWS = ["uuid", "type_name", "bid", "meta"]
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


class MobTypeNode(MemoryNode):
    TABLE_ROWS = ["uuid", "type_name", "bid", "meta"]
    TABLE = "MobTypes"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        type_name, b, m = self.agent_memory._db_read(
            "SELECT type_name, bid, meta FROM MobTypes WHERE uuid=?", self.memid
        )
        self.type_name = type_name
        self.b = b
        self.m = m

    @classmethod
    def create(cls, memory, type_name: str, idm: IDM) -> str:
        memid = cls.new(memory)
        memory._db_write(
            "INSERT INTO MobTypes(uuid, type_name, bid, meta) VALUES (?, ?, ?, ?)",
            memid,
            type_name,
            idm[0],
            idm[1],
        )
        return memid


class DanceNode(MemoryNode):
    TABLE_ROWS = ["uuid"]
    TABLE = "Dances"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        # TODO put in DB/pickle like tasks?
        self.dance_fn = self.agent_memory[memid]

    @classmethod
    def create(cls, memory, dance_fn, name=None, tags=[]) -> str:
        memid = cls.new(memory)
        memory._db_write("INSERT INTO Dances(uuid) VALUES (?)", memid)
        # TODO put in db via pickle like tasks?
        memory.dances[memid] = dance_fn
        if name is not None:
            memory.add_triple(memid, "has_name", name)
        if len(tags) > 0:
            for tag in tags:
                memory.add_triple(memid, "has_tag", tag)
        return memid


class RewardNode(MemoryNode):
    TABLE_ROWS = ["uuid", "value", "time"]
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


NODELIST = NODELIST + [
    RewardNode,
    DanceNode,
    BlockTypeNode,
    SchematicNode,
    MobNode,
    InstSegNode,
    ComponentObjectNode,
    BlockObjectNode,
]  # noqa

"""
NODELIST["rewards"] = RewardNode
NODELIST["Dances"] = DanceNode
NODELIST["BlockTypes"] = BlockTypeNode
NODELIST["Schematics"] = SchematicNode
NODELIST["Mobs"] = MobNode
NODELIST["InstSeg"] = InstSegNode
NODELIST["ComponentObjects"] = ComponentObjectNode
NODELIST["BlockObjects"] = BlockObjectNode
"""
