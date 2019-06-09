import time

class Stopwatch(object):

    def __init__(self, should_start=False):
        self.reset()
        if should_start:
            self.start()

    def start(self):
        self.t1 = time.time()

    def stop(self):
        self.t2 = time.time()

    def reset(self):
        self.t1 = 0
        self.t2 = 0

    @property
    def elapsed_time(self):
        return self.t2 - self.t1
