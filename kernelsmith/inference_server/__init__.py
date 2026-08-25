"""The live inference server: model loading, hot-swap mechanics, FastAPI endpoints.

Nothing is re-exported here. `server.py` builds `app` at import time but loads the
model only in its lifespan handler, so importing this package on a GPU-less machine
(CI) is safe.
"""
