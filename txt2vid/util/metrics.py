import collections

class RollingAvgLoss:

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.loss_window = collections.deque()
        #self.cumsum = 0.0

    def update(self, loss):
        self.loss_window.append(loss)
        #self.cumsum += loss
        if len(self.loss_window) > self.window_size:
            #self.cumsum -= self.loss_window[0]
            temp = self.loss_window[0]
            self.loss_window.popleft()
            del temp

    def get(self):
        assert len(self.loss_window) != 0
        assert len(self.loss_window) <= self.window_size

        avg = sum(self.loss_window) / len(self.loss_window)
        return avg

