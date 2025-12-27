'''
Utility class for maintaining a sliding window of recent values.
'''
from collections import deque

class SlidingWindow:
    def __init__(self, size):
        self.size = size
        self.values = deque(maxlen=size)

    def add(self, value):
        self.values.append(value)

    def average(self):
        if len(self.values) == 0:
            return 0.0
        return sum(self.values) / len(self.values)