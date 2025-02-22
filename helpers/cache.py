from collections import OrderedDict
from typing import Optional, ItemsView


# Singleton pattern
class LRUCache:
    _instance: Optional["LRUCache"] = None

    def __new__(cls, max_size: int = 5) -> "LRUCache":
        if cls._instance is None:
            cls._instance = super(LRUCache, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_size: int = 5) -> None:
        if not self._initialized:
            self.cache: OrderedDict[str, int] = OrderedDict()
            self.max_sizes: dict[str, int] = {}  # Store max size per key
            self.max_size: int = max_size
            self._initialized = True

    def set_max_size(self, key: str, max_size: int) -> None:
        """Set max size for a specific key."""
        self.max_sizes[key] = max_size

    def get(self, key: str) -> Optional[int]:
        if key in self.cache:
            self.cache.move_to_end(key)  # Mark as recently used
            return self.cache[key]
        return None

    def put(self, key: str, value: int) -> None:
        max_size: int = self.max_sizes.get(key, self.max_size)

        if key in self.cache:
            self.cache.move_to_end(key)  # Mark as recently used
        elif len(self.cache) >= max_size:
            self.cache.popitem(last=False)  # Remove oldest item

        self.cache[key] = value

    def clear(self, key: Optional[str] = None) -> None:
        """Clear all items in the cache or a specific key if provided."""
        if key:
            self.cache.pop(key, None)
            self.max_sizes.pop(key, None)
        else:
            self.cache.clear()
            self.max_sizes.clear()

    def __len__(self) -> int:
        """Return number of items in cache."""
        return len(self.cache)

    def items(self) -> ItemsView[str, int]:
        """Return a view of the cache's items (key-value pairs)."""
        return self.cache.items()

    def __iter__(self):
        """Allow iteration over the cache keys."""
        return iter(self.cache)


# Example Usage
if __name__ == "__main__":
    cache = LRUCache()
    cache.set_max_size("A", 3)
    cache.set_max_size("B", 2)

    cache.put("A", 1)
    cache.put("B", 2)
    cache.put("A", 3)
    cache.put("B", 4)
    cache.put("A", 5)
    cache.put("A", 6)  # Should evict older items if max_size reached

    print(dict(cache.items()))  # Observe cache contents
