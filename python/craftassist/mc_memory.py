"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import random
import sys
from typing import Optional, List, Tuple

from build_utils import npy_to_blocks_list
import minecraft_specs
import dance

BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(BASE_AGENT_ROOT)


from base_agent.util import XYZ, Block

from base_agent.sql_memory import AgentMemory

from base_agent.memory_nodes import (  # noqa
    TaskNode,
    PlayerNode,
    MemoryNode,
    ChatNode,
    TimeNode,
    LocationNode,
    SetNode,
    ReferenceObjectNode,
)

from mc_memory_nodes import (  # noqa
    DanceNode,
    BlockTypeNode,
    MobTypeNode,
    ObjectNode,
    BlockObjectNode,
    ComponentObjectNode,
    MobNode,
    InstSegNode,
    SchematicNode,
    NODELIST,
)

from word_maps import SPAWN_OBJECTS

BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "..")

SCHEMAS = [
    os.path.join(os.path.join(BASE_AGENT_ROOT, "base_agent"), "base_memory_schema.sql"),
    os.path.join(os.path.dirname(__file__), "mc_memory_schema.sql"),
]

SCHEMA = os.path.join(os.path.dirname(__file__), "memory_schema.sql")

THROTTLING_TICK_UPPER_LIMIT = 64
THROTTLING_TICK_LOWER_LIMIT = 4

# TODO "snapshot" memory type  (giving a what mob/object/player looked like at a fixed timestamp)
# TODO when a memory is removed, its last state should be snapshotted to prevent tag weirdness


class MCAgentMemory(AgentMemory):
    def __init__(
        self,
        db_file=":memory:",
        db_log_path=None,
        schema_paths=SCHEMAS,
        load_minecraft_specs=True,
        load_block_types=True,
        load_mob_types=True,
    ):
        super(MCAgentMemory, self).__init__(
            db_file=db_file, schema_paths=schema_paths, db_log_path=db_log_path, nodelist=NODELIST
        )
        self.banned_default_behaviors = []  # FIXME: move into triple store?
        self._safe_pickle_saved_attrs = {}
        self._load_schematics(load_minecraft_specs)
        self._load_block_types(load_block_types)
        self._load_mob_types(load_mob_types)

        self.dances = {}
        dance.add_default_dances(self)

    ###############
    ### Blocks  ###
    ###############

    def _upsert_block(
        self,
        block: Block,
        memid: str,
        table: str,
        player_placed: bool = False,
        agent_placed: bool = False,
    ):
        ((x, y, z), (b, m)) = block
        if self._db_read_one("SELECT * FROM {} WHERE x=? AND y=? AND z=?".format(table), x, y, z):
            self._db_write_blockobj(block, memid, table, player_placed, agent_placed, update=True)
        else:
            self._db_write_blockobj(block, memid, table, player_placed, agent_placed, update=False)

    def _db_write_blockobj(
        self,
        block: Block,
        memid: str,
        table: str,
        player_placed: bool,
        agent_placed: bool = False,
        update: bool = False,
    ):
        (x, y, z), (b, m) = block
        if update:
            cmd = "UPDATE {} SET uuid=?, bid=?, meta=?, updated=?, player_placed=?, agent_placed=? WHERE x=? AND y=? AND z=? "
        else:
            cmd = "INSERT INTO {} (uuid, bid, meta, updated, player_placed, agent_placed, x, y, z) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        self._db_write(
            cmd.format(table), memid, b, m, self.get_time(), player_placed, agent_placed, x, y, z
        )

    ######################
    ###  BlockObjects  ###
    ######################

    # rename this... "object" is bad name
    # also the sanity checks seem excessive?
    def get_object_by_id(self, memid: str, table="BlockObjects") -> "ObjectNode":
        # sanity check...
        r = self._db_read(
            "SELECT x, y, z, bid, meta FROM {} WHERE uuid=? ORDER BY updated".format(table), memid
        )
        assert r, memid
        # end sanity check
        if table == "BlockObjects":
            return BlockObjectNode(self, memid)
        elif table == "ComponentObjects":
            return ComponentObjectNode(self, memid)
        else:
            raise ValueError("Bad table={}".format(table))

    # and rename this
    def get_object_info_by_xyz(self, xyz: XYZ, table: str, just_memid=True):
        r = self._db_read(
            "SELECT DISTINCT(uuid), bid, meta FROM {} WHERE x=? AND y=? AND z=?".format(table),
            *xyz,
        )
        if just_memid:
            return [memid for (memid, bid, meta) in r]
        else:
            return r

    # WARNING: these do not search archived/snapshotted block objects
    # TODO replace all these all through the codebase with generic counterparts
    def get_block_object_ids_by_xyz(self, xyz: XYZ, table="BlockObjects") -> List[str]:
        return self.get_object_info_by_xyz(xyz, "BlockObjects")

    def get_block_object_by_xyz(self, xyz: XYZ) -> Optional["ObjectNode"]:
        memids = self.get_block_object_ids_by_xyz(xyz)
        if len(memids) == 0:
            return None
        return self.get_block_object_by_id(memids[0])

    # TODO remove this?
    def get_block_objects_with_tags(self, *tags) -> List["ObjectNode"]:
        tags += ("_block_object",)
        return self.get_objects_by_tags("BlockObjects", *tags)

    def get_block_object_by_id(self, memid: str) -> "ObjectNode":
        return self.get_object_by_id(memid, "BlockObjects")

    def tag_block_object_from_schematic(self, block_object_memid: str, schematic_memid: str):
        self.add_triple(block_object_memid, "_from_schematic", schematic_memid)

    # does not search archived mems for now
    # TODO remove this? replace with get_memory_by_tags(self, *tags, table=None)
    def get_objects_by_tags(self, table, *tags) -> List["ObjectNode"]:
        memids = set.intersection(*[set(self.get_memids_by_tag(t)) for t in tags])
        objects = [self.get_object_by_id(memid, table) for memid in memids]
        return [o for o in objects if o is not None]

    #######################
    ### ComponentObject ###
    #######################

    def get_component_object_by_id(self, memid: str) -> Optional["ObjectNode"]:
        return self.get_object_by_id(memid, "ComponentObjects")

    def get_component_object_ids_by_xyz(self, xyz: XYZ) -> List[str]:
        return self.get_object_info_by_xyz(xyz, "ComponentObjects")

    # TODO remove this?
    def get_component_objects_with_tags(self, *tags) -> List["ObjectNode"]:
        tags += ("_component_object",)
        return self.get_objects_by_tags("ComponentObjects", *tags)

    #####################
    ### InstSegObject ###
    #####################

    # HERE
    # TODO remove this? use get_tagged_memory(self, *tags, table=None, intersect=True)
    def get_instseg_objects_with_tags(self, *tags) -> List["ReferenceObjectNode"]:
        tags += ("_inst_seg",)
        memids = set.intersection(*[set(self.get_memids_by_tag(t)) for t in tags])
        mems = [self.get_mem_by_id(memid) for memid in memids]
        return [m for m in mems if type(m) == InstSegNode]  # noqa: T484

    ####################
    ###  Schematics  ###
    ####################

    def get_schematic_by_id(self, memid: str) -> "SchematicNode":
        return SchematicNode(self, memid)

    def get_schematic_by_property_name(self, name, table_name) -> Optional["SchematicNode"]:
        r = self._db_read(
            """
                    SELECT {}.type_name
                    FROM {} INNER JOIN Triples as T ON T.subj={}.uuid
                    WHERE (T.pred="has_name" OR T.pred="has_tag") AND T.obj=?""".format(
                table_name, table_name, table_name
            ),
            name,
        )
        if not r:
            return None

        result = []  # noqa
        for e in r:
            schematic_name = e[0]
            schematics = self._db_read(
                """
                    SELECT Schematics.uuid
                    FROM Schematics INNER JOIN Triples as T ON T.subj=Schematics.uuid
                    WHERE (T.pred="has_name" OR T.pred="has_tag") AND T.obj=?""",
                schematic_name,
            )
            if schematics:
                result.extend(schematics)
        if result:
            return self.get_schematic_by_id(random.choice(result)[0])
        else:
            return None

    def get_mob_schematic_by_name(self, name: str) -> Optional["SchematicNode"]:
        return self.get_schematic_by_property_name(name, "MobTypes")

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
        # if no schematic with exact matched name exists, search for a schematic
        # with matched property name instead
        else:
            return self.get_schematic_by_property_name(name, "BlockTypes")

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
            for premem in minecraft_specs.get_schematics():
                npy = premem["schematic"]
                memid = SchematicNode.create(self, npy_to_blocks_list(npy))
                if premem.get("name"):
                    for n in premem["name"]:
                        self.add_triple(memid, "has_name", n)
                if premem.get("tags"):
                    for t in premem["tags"]:
                        self.add_triple(memid, "has_name", t)

        # load single blocks as schematics
        bid_to_name = minecraft_specs.get_block_data()["bid_to_name"]
        for (d, m), name in bid_to_name.items():
            if d >= 256:
                continue
            memid = SchematicNode.create(self, [((0, 0, 0), (d, m))])
            self.add_triple(memid, "has_name", name)
            if "block" in name:
                self.add_triple(memid, "has_name", name.strip("block").strip())

    def _load_block_types(
        self,
        load_block_types=True,
        load_color=True,
        load_block_property=True,
        simple_color=False,
        load_material=True,
    ):
        if not load_block_types:
            return
        bid_to_name = minecraft_specs.get_block_data()["bid_to_name"]

        color_data = minecraft_specs.get_colour_data()
        if simple_color:
            name_to_colors = color_data["name_to_simple_colors"]
        else:
            name_to_colors = color_data["name_to_colors"]

        block_property_data = minecraft_specs.get_block_property_data()
        block_name_to_properties = block_property_data["name_to_properties"]

        for (b, m), type_name in bid_to_name.items():
            if b >= 256:
                continue
            memid = BlockTypeNode.create(self, type_name, (b, m))
            self.add_triple(memid, "has_name", type_name)
            if "block" in type_name:
                self.add_triple(memid, "has_name", type_name.strip("block").strip())

            if load_color:
                if name_to_colors.get(type_name) is not None:
                    for color in name_to_colors[type_name]:
                        self.add_triple(memid, "has_colour", color)

            if load_block_property:
                if block_name_to_properties.get(type_name) is not None:
                    for property in block_name_to_properties[type_name]:
                        self.add_triple(memid, "has_name", property)

    def _load_mob_types(self, load_mob_types=True):
        if not load_mob_types:
            return

        mob_property_data = minecraft_specs.get_mob_property_data()
        mob_name_to_properties = mob_property_data["name_to_properties"]
        for (name, m) in SPAWN_OBJECTS.items():
            type_name = "spawn " + name

            # load single mob as schematics
            memid = SchematicNode.create(self, [((0, 0, 0), (383, m))])
            self.add_triple(memid, "has_name", type_name)
            if "block" in type_name:
                self.add_triple(memid, "has_name", type_name.strip("block").strip())

            # then load properties
            memid = MobTypeNode.create(self, type_name, (383, m))
            self.add_triple(memid, "has_name", type_name)
            if mob_name_to_properties.get(type_name) is not None:
                for property in mob_name_to_properties[type_name]:
                    self.add_triple(memid, "has_name", property)

    ##############
    ###  Mobs  ###
    ##############

    # does not return archived mems
    def get_mobs(
        self,
        spatial_range: Tuple[int, int, int, int, int, int] = None,
        player_placed: bool = None,
        agent_placed: bool = None,
        spawntime: Tuple[int, int] = None,
        mobtype: str = None,
    ) -> List["MobNode"]:
        """Find mobs matching the given filters

        Args:
          spatial_range = [xmin, xmax, ymin, ymax, zmin, zmax]
          spawntime = [time_min, time_max] .  can be negative to ignore
              spawntime is in the form of self.get_time()
          mobtype = string
        """

        def maybe_add_AND(query):
            if query[-6:] != "WHERE ":
                query += " AND"
            return query

        query = "SELECT MOBS.uuid FROM Mobs INNER JOIN Memories as M ON Mobs.uuid=M.uuid WHERE M.is_snapshot=0 AND"
        #        query = "SELECT uuid FROM Mobs WHERE "
        args: List = []
        if spatial_range is not None:
            args.extend(spatial_range)
            query += " x>? AND x<? AND y>? AND y<? AND z>? AND z<?"
        if player_placed is not None:
            args.append(player_placed)
            query = maybe_add_AND(query) + " player_placed = ? "
        if agent_placed is not None:
            args.append(agent_placed)
            query = maybe_add_AND(query) + " agent_placed = ? "
        if spawntime is not None:
            if spawntime[0] > 0:
                args.append(spawntime[0])
                query = maybe_add_AND(query) + " spawn >= ? "
            if spawntime[1] > 0:
                args.append(spawntime[1])
                query = maybe_add_AND(query) + " spawn <= ? "
        if mobtype is not None:
            args.append(mobtype)
            query = maybe_add_AND(query) + " mobtype=?"
        memids = [m[0] for m in self._db_read(query, *args)]
        return [self.get_mob_by_id(memid) for memid in memids]

    # does not return archived mems
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
            (memid,) = r
        else:
            memid = MobNode.create(self, mob)
        return self.get_mob_by_id(memid)

    ###############
    ###  Dances  ##
    ###############

    def add_dance(self, dance_fn, name=None, tags=[]):
        # a dance is movement determined as a sequence of steps, rather than by its destination
        return DanceNode.create(self, dance_fn, name=name, tags=tags)
