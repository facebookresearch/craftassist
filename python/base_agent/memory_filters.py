from typing import List

SELFID = "0" * 32


def maybe_and(sql, a):
    if a:
        return sql + " AND "
    else:
        return sql


def maybe_or(sql, a):
    if a:
        return sql + " OR "
    else:
        return sql


# TODO counts
def get_property_value(agent_memory, mem, prop):
    # order of precedence:
    # 1: main memory table
    # 2: table corresponding to the nodes .TABLE
    # 3: triple with the nodes memid as subject and prop as predicate

    # is it in the main memory table?
    cols = [c[1] for c in agent_memory._db_read("PRAGMA table_info(Memories)")]
    if prop in cols:
        cmd = "SELECT " + prop + " FROM Memories WHERE uuid=?"
        r = agent_memory._db_read(cmd, mem.memid)
        return r[0][0]
    # is it in the mem.TABLE?
    T = mem.TABLE
    cols = [c[1] for c in agent_memory._db_read("PRAGMA table_info({})".format(T))]
    if prop in cols:
        cmd = "SELECT " + prop + " FROM " + T + " WHERE uuid=?"
        r = agent_memory._db_read(cmd, mem.memid)
        return r[0][0]
    # is it a triple?
    triples = agent_memory.get_triples(subj=mem.memid, pred_text=prop, return_obj_text="always")
    if len(triples) > 0:
        return triples[0][2]

    return None


class MemorySearcher:
    def __init__(self, self_memid=SELFID, search_data=None):
        self.self_memid = self_memid
        self.search_data = search_data

    def search(self, memory, search_data=None) -> List["ReferenceObjectNode"]:  # noqa T484
        raise NotImplementedError


class ReferenceObjectSearcher(MemorySearcher):
    def __init__(self, self_memid=SELFID, search_data=None):
        super().__init__(self_memid=SELFID, search_data=None)

    def is_filter_empty(self, filter_dict):
        r = filter_dict.get("special")
        if r and len(r) > 0:
            return False
        r = filter_dict.get("ref_obj_range")
        if r and len(r) > 0:
            return False
        r = filter_dict.get("ref_obj_exact")
        if r and len(r) > 0:
            return False
        r = filter_dict.get("memories_range")
        if r and len(r) > 0:
            return False
        r = filter_dict.get("memories_exact")
        if r and len(r) > 0:
            return False
        t = filter_dict.get("triples")
        if t and len(t) > 0:
            return False
        return True

    def range_queries(self, r, table, a=False):
        """ this does x, y, z, pitch, yaw, etc.
        input format for generates is 
        {"xmin": float, xmax: float, ... , yawmin: float, yawmax: float}
        """
        sql = ""
        vals = []
        for k, v in r.items():
            if "min" in k:
                sql = maybe_and(sql, len(vals) > 0)
                sql += table + "." + k.replace("min", "") + ">? "
                vals.append(v)
            if "max" in k:
                sql = maybe_and(sql, len(vals) > 0)
                sql += table + "." + k.replace("max", "") + "<? "
                vals.append(v)
        return sql, vals

    def exact_matches(self, m, table, a=False):
        sql = ""
        vals = []
        for k, v in m.items():
            sql = maybe_and(sql, len(vals) > 0)
            sql += table + "." + k + "=? "
            vals.append(v)
        return sql, vals

    def triples(self, triples, a=False):
        # currently does an "and": the memory needs to satisfy all triples
        vals = []
        if not triples:
            return "", vals
        sql = "ReferenceObjects.uuid IN (SELECT subj FROM Triples WHERE "
        for t in triples:
            sql = maybe_or(sql, len(vals) > 0)
            vals.append(t["pred_text"])
            if t.get("obj_text"):
                sql += "(pred_text, obj_text)=(?, ?)"
                vals.append(t["obj_text"])
            else:
                sql += "(pred_text, obj)=(?, ?)"
                vals.append(t["obj"])
        sql += " GROUP BY subj HAVING COUNT(subj)=? )"
        vals.append(len(triples))
        return sql, vals

    def get_query(self, filter_dict, ignore_self=True):
        if self.is_filter_empty(filter_dict):
            query = "SELECT uuid FROM ReferenceObjects"
            if ignore_self:
                query += " WHERE uuid !=?"
                return query, [self.self_memid]
            else:
                return query, []

        query = (
            "SELECT ReferenceObjects.uuid FROM ReferenceObjects"
            " INNER JOIN Memories as M on M.uuid=ReferenceObjects.uuid"
            " WHERE "
        )

        args = []
        fragment, vals = self.range_queries(
            filter_dict.get("ref_obj_range", {}), "ReferenceObjects"
        )
        query = maybe_and(query, len(args) > 0)
        args.extend(vals)
        query += fragment

        fragment, vals = self.exact_matches(
            filter_dict.get("ref_obj_exact", {}), "ReferenceObjects"
        )
        query = maybe_and(query, len(args) > 0 and len(vals) > 0)
        args.extend(vals)
        query += fragment

        fragment, vals = self.range_queries(filter_dict.get("memories_range", {}), "M")
        query = maybe_and(query, len(args) > 0 and len(vals) > 0)
        args.extend(vals)
        query += fragment

        fragment, vals = self.exact_matches(filter_dict.get("memories_exact", {}), "M")
        query = maybe_and(query, len(args) > 0 and len(vals) > 0)
        args.extend(vals)
        query += fragment

        fragment, vals = self.triples(filter_dict.get("triples", []))
        query = maybe_and(query, len(args) > 0 and len(vals) > 0)
        args.extend(vals)
        query += fragment

        if ignore_self:
            query += " AND ReferenceObjects.uuid !=?"
            args.append(self.self_memid)
        return query, args

    # flag (default) so that it makes a copy of speaker_look etc so that if the searcher is called
    # later so it doesn't return the new position of the agent/speaker/speakerlook
    # how to parse this distinction?
    def handle_special(self, memory, search_data):
        d = search_data.get("special")
        if not d:
            return []
        if d.get("SPEAKER"):
            return [memory.get_player_by_eid(d["SPEAKER"])]
        if d.get("SPEAKER_LOOK"):
            memids = memory._db_read_one(
                'SELECT uuid FROM ReferenceObjects WHERE ref_type="attention" AND type_name=?',
                d["SPEAKER_LOOK"],
            )
            if memids:
                memid = memids[0]
                mem = memory.get_location_by_id(memid)
                return [mem]
        if d.get("AGENT"):
            return [memory.get_player_by_eid(d["AGENT"])]
        if d.get("DUMMY"):
            return [d["DUMMY"]]
        return []

    def search(self, memory, search_data=None) -> List["ReferenceObjectNode"]:  # noqa T484
        """Find ref_objs matching the given filters
        filter_dict has children:
            "ref_obj_range", dict, with keys "min<column_name>" or "max<column_name>", 
                  (that is the string "min" prepended to the column name)
                  and float values vmin and vmax respectively.  
                  <column_name> is any column in the ReferenceObjects table that
                  is a numerical value.  filters on rows satisfying the inequality 
                  <column_entry> > vmin or <column_entry> < vmax
            "ref_obj_exact", dict,  with keys "<column_name>"
                  <column_name> is any column in the ReferenceObjects table
                  checks exact matches to the value
            "memories_range" and "memories_exact" are the same, but columns in the Memories table
            "triples" list [t0, t1, ...,, tm].  each t in the list is a dict
                  with form t = {"pred_text": <pred>, "obj_text": <obj>}
                  or t = {"pred_text": <pred>, "obj": <obj_memid>}
                  currently returns memories with all triples matched 
        """
        if not search_data:
            search_data = self.search_data
        assert search_data
        if search_data.get("special"):
            return self.handle_special(memory, search_data)
        query, args = self.get_query(search_data)
        self.search_data = search_data
        memids = [m[0] for m in memory._db_read(query, *args)]
        return [memory.get_mem_by_id(memid) for memid in memids]


if __name__ == "__main__":
    filter_dict = {
        "ref_obj_range": {"minx": 3},
        "memories_exact": {"create_time": 1},
        "triples": [
            {"pred_text": "has_tag", "obj_text": "cow"},
            {"pred_text": "has_name", "obj_text": "eddie"},
        ],
    }
