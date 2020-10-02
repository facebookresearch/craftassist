"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
from memory_filters import ReferenceObjectSearcher, get_property_value
from base_util import TICKS_PER_SEC, TICKS_PER_MINUTE, TICKS_PER_HOUR


# attribute has function signature list(mems) --> list(float)
class Attribute:
    def __init__(self, agent):
        self.agent = agent

    def __call__(self, mems):
        raise NotImplementedError("Implemented by subclass")


class TableColumn(Attribute):
    def __init__(self, agent, attribute):
        super().__init__(agent)
        self.attribute = attribute

    def __call__(self, mems):
        return [get_property_value(self.agent.memory, mem, self.attribute) for mem in mems]


class LinearExtentAttribute(Attribute):
    """ 
    computes the (perhaps signed) length between two points in space.  
    if "relative_direction"=="AWAY", unsigned length
    if "relative_direction" in ["LEFT", "RIGHT" ...] projected onto a special direction 
         and signed.  the "arrow" goes from "source" to "destination",
         e.g. if destination is more LEFT than source, "LEFT" will be positive
    if "relative_direction" in ["INSIDE", "OUTSIDE"], signed length is shifted towards zero
         so that 0 is at the boundary of the source.
         This is not implemented yet FIXME!!
    One of the two points in space is given by the positions of a reference object
    either given directly as a memory, or given as FILTERs to search
    the other is the list element input into the call
    """

    def __init__(self, agent, location_data, mem=None, fixed_role="source"):
        super().__init__(agent)
        self.coordinate_transforms = agent.coordinate_transforms
        self.location_data = location_data
        self.fixed_role = fixed_role

        self.frame = location_data.get("frame") or "AGENT"

        # TODO generalize/formalize this
        # TODO: currently stores look vecs/orientations at creation,
        #     build mechanism to update orientations, e.g. if giving directions
        #     "first you turn left, then go 7 steps forward, turn right, go 7 steps forward"
        #     need this in grammar too
        # TODO store fixed pitch/yaw etc. with arxiv memories, not raw
        try:
            if self.frame == "AGENT":
                # TODO handle this appropriately!
                yaw, pitch = agent.memory._db_read(
                    "SELECT yaw, pitch FROM ReferenceObjects WHERE uuid=?", agent.memory.self_memid
                )[0]
            elif self.frame == "ABSOLUTE":
                yaw, pitch = self.coordinate_transforms.yaw_pitch(
                    self.coordinate_transforms.DIRECTIONS["FRONT"]
                )
            # this is another player/agent; it is assumed that the frame has been replaced with
            # with the eid of the player/agent
            else:
                # TODO error if eid not found; but then parent/helper should have caught it?
                # TODO error properly if eid is a ref object, but pitch or yaw are null
                yaw, pitch = agent.memory._db_read(
                    "SELECT yaw, pitch FROM ReferenceObjects WHERE eid=?", self.frame
                )[0]
        except:
            # TODO handle this better
            raise Exception(
                "Unable to find the yaw, pitch in the given frame; maybe can't find the eid?"
            )

        self.yaw = yaw
        self.pitch = pitch
        self.mem = mem
        self.searcher = "mem"
        # put a "NULL" mem in input to not build a searcher
        if not self.mem:
            d = self.location_data.get(fixed_role)
            if d:
                self.searchers = ReferenceObjectSearcher(search_data=d)

    def extent(self, source, destination):
        # source and destination are arrays in this function
        # arrow goes from source to destination:
        diff = np.subtract(source, destination)
        if self.location_data["relative_direction"] in ["INSIDE", "OUTSIDE"]:
            raise Exception("inside and outside not yet implemented in linear extent")
        if self.location_data["relative_direction"] in [
            "LEFT",
            "RIGHT",
            "UP",
            "DOWN",
            "FRONT",
            "BACK",
        ]:
            reldir_vec = self.coordinate_transforms.DIRECTIONS[
                self.location_data["relative_direction"]
            ]
            # this should be an inverse transform so we set inverted=True
            dir_vec = self.coordinate_transforms.transform(
                reldir_vec, self.yaw, self.pitch, inverted=True
            )
            return diff @ dir_vec
        else:  # AWAY
            return np.linalg.norm(diff)

    def __call__(self, mems):
        if not self.mem:
            fixed_mem = self.searcher.search(self.agent.memory)
            # FIXME!!! handle mem not found
        else:
            fixed_mem = self.mem
        # FIMXE TODO store and use an arxiv if we don't want position to track!
        if self.fixed_role == "source":
            return [self.extent(fixed_mem.get_pos(), mem.get_pos()) for mem in mems]
        else:
            return [self.extent(mem.get_pos(), fixed_mem.get_pos()) for mem in mems]


# a value has a get_value() method; and get_value should not have
# any inputs
class ComparisonValue:
    def __init__(self, agent):
        self.agent = agent

    def get_value(self):
        raise NotImplementedError("Implemented by subclass")


# TODO more composable less ugly, more ML friendly
class ScaledValue(ComparisonValue):
    def __init__(self, value, scale):
        self.value = value
        self.scale = scale

    def get_value(self):
        return self.scale * self.value.get_value()


# TODO feet, meters, inches, centimeters, degrees, etc.
# each of these converts a measure into the agents internal units,
#     e.g. seconds/minutes/hours to ticks
#     inches/centimeters/feet to meters or blocks (assume 1 block in mc equals 1 meter in real world)
conversion_factors = {
    "seconds": TICKS_PER_SEC,
    "minutes": TICKS_PER_MINUTE,
    "hours": TICKS_PER_HOUR,
}


def convert_comparison_value(comparison_value, unit):
    if not unit:
        return comparison_value
    assert conversion_factors.get(unit)
    return ScaledValue(comparison_value, conversion_factors[unit])


class FixedValue(ComparisonValue):
    def __init__(self, agent, value):
        super().__init__(agent)
        self.value = value

    def get_value(self):
        return self.value


# TODO store more in memory,
# or at least
# make some TimeNodes as side effects
# WARNING:  elapsed mode uses get_time at construction as 0
class TimeValue(ComparisonValue):
    """ 
    modes are elapsed, time, and world_time.
    if "elapsed" or "time" uses memory.get_time as timer
    if "elapsed", value is offset by time at creation
    if "world_time" uses memory.get_world_time
    """

    def __init__(self, agent, mode="elapsed"):
        self.mode = mode
        self.offset = 0.0
        if self.mode == "elapsed":
            self.offset = agent.memory.get_time()
            self.get_time = agent.memory.get_time
        elif self.mode == "time":
            self.get_time = agent.memory.get_time
        else:  # world_time
            self.get_time = agent.memory.get_world_time

    def get_value(self):
        return self.get_time() - self.offset


# TODO unit conversions?
class MemoryColumnValue(ComparisonValue):
    def __init__(self, agent, search_data, mem=None):
        super().__init__(agent)
        self.search_data = search_data
        # TODO expand beyond ref objects
        self.mem = mem
        if not self.mem:
            self.searcher = ReferenceObjectSearcher(search_data=search_data)

    def get_value(self):
        if self.mem:
            return self.search_data["attribute"]([self.mem])[0]
        mems = self.searcher.search(self.agent.memory)
        if len(mems) > 0:
            # TODO/FIXME! deal with more than 1 better
            return self.search_data["attribute"](mems)[0]
        else:
            return


class LinearExtentValue(ComparisonValue):
    # this is a linear extent with both source and destination filled.
    # e.g. "when you are as far from the house as the cow is from the house"
    # but NOT for "when the cow is 3 steps from the house"
    # in the latter case, one of the two entities will be given by the filters
    def __init__(self, agent, linear_exent_attribute, mem=None, search_data=None):
        super().__init__(agent)
        self.linear_extent_attribute = linear_exent_attribute
        assert mem or search_data
        self.searcher = None
        self.mem = mem
        if not self.mem:
            self.searcher = ReferenceObjectSearcher(search_data=search_data)

    def get_value(self):
        if self.mem:
            mems = [self.mem]
        else:
            mems = self.searcher.search(self.agent.memory)
        if len(mems) > 0:
            # TODO/FIXME! deal with more than 1 better
            return self.linear_extent_attribute(mems)[0]
        else:
            return


class Condition:
    def __init__(self, agent):
        self.agent = agent

    def check(self) -> bool:
        raise NotImplementedError("Implemented by subclass")


class NeverCondition(Condition):
    def __init__(self, agent):
        super().__init__(agent)
        self.name = "never"

    def check(self):
        return False


class AndCondition(Condition):
    """ conditions should be an iterable"""

    def __init__(self, agent, conditions):
        super().__init__(agent)
        self.name = "and"
        self.conditions = conditions

    def check(self):
        for c in self.conditions:
            if not c.check():
                return False
        return True


class OrCondition(Condition):
    """ conditions should be an iterable"""

    def __init__(self, agent, conditions):
        super().__init__(agent)
        self.name = "or"
        self.conditions = conditions

    def check(self):
        for c in self.conditions:
            if c.check():
                return True
        return False


# start_time and end_time are in (0, 1)
# 0 is sunrise, .5 is sunset
def build_special_time_condition(agent, start_time, end_time, epsilon=0.01):
    value_left = TimeValue(agent, mode="world_time")
    if end_time > 0:
        start = Comparator(
            comparison_type="GREATER_THAN_EQUAL", value_left=value_left, value_right=start_time
        )
        end = Comparator(
            comparison_type="LESS_THAN_EQUAL", value_left=value_left, value_right=end_time
        )
        return AndCondition(agent, [start, end])
    else:
        return Comparator(
            comparison_type="CLOSE_TO",
            value_left=value_left,
            value_right=start_time,
            epsilon=epsilon,
        )


# TODO make this more ML friendly?
# eventually do "x minutes before condition"?  how?
# KEEPS state (did the event occur- starts timer then)
class TimeCondition(Condition):
    """ 
    if event is None, the timer starts now
    if event is not None, it should be a condition, timer starts on the condition being true
    This time condition is true when the comparator between 
    timer (as value_left) and the comparator's value_right is true
    if comparator is a string, it should be "SUNSET" / "SUNRISE" / "DAY" / "NIGHT" / "AFTERNOON" / "MORNING"
    else it should be built in the parent, and the value_right should be commeasurable (properly scaled)
    """

    def __init__(self, agent, comparator, event=None):
        super().__init__(agent)
        self.special = None
        self.event = event
        if type(comparator) is str:
            if comparator == "SUNSET":
                self.special = build_special_time_condition(agent, 0.5, -1)
            elif comparator == "SUNRISE":
                self.special = build_special_time_condition(agent, 0.0, -1)
            elif comparator == "MORNING":
                self.special = build_special_time_condition(agent, 0, 0.25)
            elif comparator == "AFTERNOON":
                self.special = build_special_time_condition(agent, 0.25, 0.5)
            elif comparator == "DAY":
                self.special = build_special_time_condition(agent, 0.0, 0.5)
            elif comparator == "NIGHT":
                self.special = build_special_time_condition(agent, 0.5, 1.0)
            else:
                raise NotImplementedError("unknown special time condition type: " + comparator)
        else:
            if not event:
                comparator.value_left = TimeValue(agent, mode="elapsed")
            self.comparator = comparator

    def check(self):
        if not self.event:
            return self.comparator.check()
        else:
            if self.event.check():
                self.comparator.value_left = TimeValue(self.agent, mode="elapsed")
                self.event = None
            return self.comparator.check()


class Comparator(Condition):
    def __init__(
        self, agent, comparison_type="EQUAL", value_left=None, value_right=None, epsilon=0
    ):
        super().__init__(agent)
        self.comparison_type = comparison_type
        self.value_left = value_left
        self.value_right = value_right
        self.epsilon = epsilon

    # raise errors if no value left or right?
    # raise errors if strings compared with > < etc.?
    # FIXME handle type mismatches
    # TODO less types, use NotCondition
    # TODO MOD_EQUAL, MOD_CLOSE
    def check(self):
        value_left = self.value_left.get_value()
        value_right = self.value_right.get_value()
        if not self.value_left:
            return False
        if not value_right:
            return False
        if self.comparison_type == "GREATER_THAN_EQUAL":
            return value_left >= value_right
        elif self.comparison_type == "GREATER_THAN":
            return value_left > value_right
        elif self.comparison_type == "EQUAL":
            return value_left == value_right
        elif self.comparison_type == "NOT_EQUAL":
            return value_left != value_right
        elif self.comparison_type == "LESS_THAN":
            return value_left < value_right
        elif self.comparison_type == "CLOSE_TO":
            return abs(value_left - value_right) <= self.epsilon
        else:
            # self.comparison_type == "LESS_THAN_EQUAL":
            return value_left <= value_right
