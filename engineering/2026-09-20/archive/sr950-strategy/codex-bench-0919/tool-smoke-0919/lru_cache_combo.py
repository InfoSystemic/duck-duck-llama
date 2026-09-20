from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        self.capacity = capacity
        self._data = OrderedDict()

    def get(self, key):
        if self.capacity == 0 or key not in self._data:
            return -1
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key, value):
        if self.capacity == 0:
            return
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)
