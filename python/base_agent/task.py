DEFAULT_THROTTLING_TICK = 16
THROTTLING_TICK_UPPER_LIMIT = 64
THROTTLING_TICK_LOWER_LIMIT = 4

# put a counter and a max_count so can't get stuck?
class Task(object):
    def __init__(self):
        self.interrupted = False
        self.finished = False
        self.name = None
        self.undone = False
        self.last_stepped_time = None
        self.throttling_tick = DEFAULT_THROTTLING_TICK

    def step(self, agent):
        return

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
