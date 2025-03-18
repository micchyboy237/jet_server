import hashlib
import os
import pickle

CACHE_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_server/.cache/heuristics"
CACHE_FILE = "ngrams_cache.pkl"  # Name of the cache file


class CacheManager:
    def __init__(self, cache_dir=CACHE_DIR, cache_file=CACHE_FILE):
        self.cache_dir = cache_dir
        self.cache_file = cache_file

    def _get_cache_path(self):
        """Return the full path of the cache file."""
        return os.path.join(self.cache_dir, self.cache_file)

    def get_file_hash(self, file_path: str) -> str:
        """Generate a hash for the given file."""
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def load_cache(self) -> dict:
        """Load the cache file if exists, otherwise return an empty dict."""
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                return pickle.load(cache_file)
        return {}

    def save_cache(self, data: dict) -> None:
        """Save the cache to a file."""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = self._get_cache_path()
        with open(cache_path, 'wb') as cache_file:
            pickle.dump(data, cache_file)

    def is_cache_valid(self, file_path: str, cache_data: dict) -> bool:
        """Check if the cache is valid by comparing file hashes."""
        current_file_hash = self.get_file_hash(file_path)
        return cache_data.get("file_hash") == current_file_hash

    def update_cache(self, file_path: str, ngrams: list) -> dict:
        """Regenerate the cache with new data."""
        current_file_hash = self.get_file_hash(file_path)
        cache_data = {
            "file_hash": current_file_hash,
            "common_texts_ngrams": ngrams
        }
        self.save_cache(cache_data)
        return cache_data
