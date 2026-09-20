from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        self.capacity = capacity
        self._entries = OrderedDict()

    def get(self, key):
        if key not in self._entries:
            return -1
        self._entries.move_to_end(key)
        return self._entries[key]

    def put(self, key, value):
        if self.capacity == 0:
            return
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = value
        if len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
