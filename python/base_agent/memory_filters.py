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


class ReferenceObjectSearcher:
    def __init__(self, self_memid=SELFID):
        self.self_memid = self_memid

    def is_filter_empty(self, filter_dict):
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

    def search(self, memory, filter_dict) -> List["ReferenceObjectNode"]:  # noqa T484
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
        query, args = self.get_query(filter_dict)
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
