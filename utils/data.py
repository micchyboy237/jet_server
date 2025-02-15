import json
import hashlib
from typing import Any


def generate_key(*args: Any) -> str:
    """
    Generate a SHA256 hash key from the concatenation of input arguments.

    Args:
        *args: Variable length argument list.

    Returns:
        A SHA256 hash string.
    """
    try:
        # Combine the arguments into a JSON string
        concatenated = json.dumps(args, separators=(',', ':'))
        # Generate a SHA256 hash of the concatenated string
        key = hashlib.sha256(concatenated.encode()).hexdigest()
        return key
    except TypeError as e:
        raise ValueError(f"Invalid argument provided: {e}")
