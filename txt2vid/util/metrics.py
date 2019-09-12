import collections

class RollingAvg:

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.window = collections.deque()
        #self.cumsum = 0.0

    def update(self, x):
        self.window.append(x)
        #self.cumsum += loss
        if len(self.window) > self.window_size:
            #self.cumsum -= self.loss_window[0]
            temp = self.window[0]
            self.window.popleft()
            del temp

    def get(self):
        assert len(self.window) != 0
        assert len(self.window) <= self.window_size

        return sum(self.window) / len(self.window)
