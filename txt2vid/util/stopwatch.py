from datetime import datetime

class Stopwatch(object):

    def __init__(self, should_start=False):
        self.reset()
        if should_start:
            self.start()

    def start(self):
        self.t1 = datetime.now()

    def stop(self):
        self.t2 = datetime.now()

    def reset(self):
        self.t1 = 0
        self.t2 = 0

    @property
    def elapsed_time(self):
        diff = self.t2 - self.t1
        return diff.microseconds / 10**6
