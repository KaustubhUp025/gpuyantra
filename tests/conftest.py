"""Test-wide setup.

kernelsmith.config reads GOOGLE_CLOUD_PROJECT strictly (os.environ[...]) so a missing
env var fails loudly in production. Unit tests never touch GCP, so provide the value
here — before any test module imports kernelsmith.config.
"""

import os

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "gpuyantra")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
