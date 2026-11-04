import os
import tempfile

# Before anything imports speasy: tests must never write into the user's Speasy cache.
os.environ["SPEASY_CACHE_PATH"] = tempfile.mkdtemp(prefix="tohban-speasy-cache-")
