from condition import NeverCondition

DEFAULT_THROTTLING_TICK = 16
THROTTLING_TICK_UPPER_LIMIT = 64
THROTTLING_TICK_LOWER_LIMIT = 4

# put a counter and a max_count so can't get stuck?
class Task(object):
    def __init__(self):
        self.memid = None
        self.interrupted = False
        self.finished = False
        self.name = None
        self.undone = False
        self.last_stepped_time = None
        self.throttling_tick = DEFAULT_THROTTLING_TICK
        self.stop_condition = NeverCondition(None)

    def step(self, agent):
        # todo? make it so something stopped by condition can be resumed?
        if self.stop_condition.check():
            self.finished = True
            return
        return

    def add_child_task(self, t, agent, pass_stop_condition=True):
        # FIXME, this is ugly and dangerous; some conditions might keep state etc?
        if pass_stop_condition:
            t.stop_condition = self.stop_condition
        agent.memory.task_stack_push(t, parent_memid=self.memid)

    def interrupt(self):
        self.interrupted = True

    def check_finished(self):
        if self.finished:
            return self.finished

    def hurry_up(self):
        self.throttling_tick /= 4
        if self.throttling_tick < THROTTLING_TICK_LOWER_LIMIT:
            self.throttling_tick = THROTTLING_TICK_LOWER_LIMIT

    def slow_down(self):
        self.throttling_tick *= 4
        if self.throttling_tick > THROTTLING_TICK_UPPER_LIMIT:
            self.throttling_tick = THROTTLING_TICK_UPPER_LIMIT

    def __repr__(self):
        return str(type(self))
