from collections import OrderedDict


class LRUCache:
    def __init__(self, max_size=5):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)  # Mark as recently used
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)  # Mark as recently used
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove oldest item
        self.cache[key] = value

    def clear(self):
        """Clear all items in the cache."""
        self.cache.clear()
