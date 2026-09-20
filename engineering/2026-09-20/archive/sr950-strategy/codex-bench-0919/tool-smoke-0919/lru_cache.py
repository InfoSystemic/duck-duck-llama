class LRUCache:
    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        self.capacity = capacity
        self._entries = {}

    def get(self, key):
        if key not in self._entries:
            return -1
        value = self._entries.pop(key)
        self._entries[key] = value
        return value

    def put(self, key, value):
        if self.capacity == 0:
            return
        if key in self._entries:
            self._entries.pop(key)
        elif len(self._entries) >= self.capacity:
            self._entries.pop(next(iter(self._entries)))
        self._entries[key] = value
