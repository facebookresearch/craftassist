from typing import Optional
from base_util import ErrorWithResponse
from condition import (
    Condition,
    NeverCondition,
    AndCondition,
    OrCondition,
    Comparator,
    MemoryColumnValue,
    FixedValue,
    TimeCondition,
    TableColumn,
)
from attribute_helper import interpret_linear_extent, interpret_span_value, maybe_specific_mem


class ConditionInterpreter:
    def __init__(self):
        # extra layer of indirection to allow easier split into base_agent and specialized agent conditions...
        self.condition_types = {
            "NEVER": self.interpret_never,
            "AND": self.interpret_and,
            "OR": self.interpret_or,
            "TIME": self.interpret_time,
            "COMPARATOR": self.interpret_comparator,
        }
        # to avoid having to redefine interpret_comparator in agents if necessary ...
        # TODO distance between
        self.value_extractors = {
            "filters": MemoryColumnValue,
            "span": FixedValue,
            "distance_between": None,
        }

    def __call__(self, interpreter, speaker, d) -> Optional[Condition]:
        ct = d.get("condition_type")
        if ct:
            if self.condition_types.get(ct):
                # condition_type NEVER doesn't have a "condition" sibling
                if ct == "NEVER":
                    return self.condition_types[ct](interpreter, speaker, d)
                if not d.get("condition"):
                    raise ErrorWithResponse(
                        "I thought there was a condition but I don't understand it"
                    )
                return self.condition_types[ct](interpreter, speaker, d["condition"])
            else:
                raise ErrorWithResponse("I don't understand that condition")
        else:
            return None

    def interpret_never(self, interpreter, speaker, d) -> Optional[Condition]:
        return NeverCondition(interpreter.agent)

    def interpret_or(self, interpreter, speaker, d) -> Optional[Condition]:
        orlist = d["or_condition"]
        conds = []
        for c in orlist:
            new_condition = self(interpreter, speaker, d)
            if new_condition:
                conds.append(new_condition)
        return OrCondition(interpreter.agent, conds)

    def interpret_and(self, interpreter, speaker, d) -> Optional[Condition]:
        orlist = d["and_condition"]
        conds = []
        for c in orlist:
            new_condition = self(interpreter, speaker, d)
            if new_condition:
                conds.append(new_condition)
        return AndCondition(interpreter.agent, conds)

    def interpret_time(self, interpreter, speaker, d):
        event = None

        if d.get("special_time_event"):
            return TimeCondition(interpreter.agent, d["special_time_event"])
        else:
            if not d.get("comparator"):
                raise ErrorWithResponse("I don't know how to interpret this time condition")
            dc = d["comparator"]
            dc["input_left"] = {"value_extractor": "NULL"}
            comparator = self.interpret_comparator(interpreter, speaker, dc)

        if d.get("event"):
            event = self(interpreter, speaker, d["event"])

        return TimeCondition(interpreter.agent, comparator, event=event)

    # TODO distance between
    # TODO make this more modular.  what if we want to redefine just distance_between in a new agent?
    def interpret_comparator(self, interpreter, speaker, d):
        input_left_d = d.get("input_left")
        input_right_d = d.get("input_right")
        if (not input_right_d) or (not input_left_d):
            return None
        value_extractors = {}
        for inp_pos in ["input_left", "input_right"]:
            inp = d[inp_pos]["value_extractor"]
            if type(inp) is str:
                if inp == "NULL":
                    value_extractors[inp_pos] = None
                else:
                    # this is a span
                    cm = d.get("comparison_measure")
                    v = interpret_span_value(interpreter, speaker, inp, comparison_measure=cm)
                    value_extractors[inp_pos] = v
            elif inp.get("output"):
                # this is a filter
                # TODO FIXME! deal with count
                # TODO logical form etc.?
                a = inp["output"]["attribute"]
                if type(a) is str:
                    search_data = {"attribute": TableColumn(interpreter.agent, a)}
                elif a.get("linear_extent"):
                    search_data = {
                        "attribute": interpret_linear_extent(
                            interpreter, speaker, a["linear_extent"]
                        )
                    }
                mem, sd = maybe_specific_mem(interpreter, speaker, {"filters": inp})
                if sd:
                    for k, v in sd.items():
                        search_data[k] = v
                # TODO wrap this in a ScaledValue using condtition.convert_comparison_value
                # and "comparison_measure"
                value_extractors[inp_pos] = MemoryColumnValue(
                    interpreter.agent, search_data, mem=mem
                )
            else:
                raise ErrorWithResponse(
                    "I don't know understand that condition, looks like a comparator but value is not filters or span"
                )
        comparison_type = d.get("comparison_type")
        if not comparison_type:
            ErrorWithResponse(
                "I think you want me to compare two things in a condition, but am not sure what type of comparison"
            )

        return Comparator(
            interpreter.agent,
            value_left=value_extractors["input_left"],
            value_right=value_extractors["input_right"],
            comparison_type=comparison_type,
        )

    # TODO not
