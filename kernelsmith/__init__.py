"""KernelSmith — an ADK agent tree that writes, verifies and hot-swaps Triton kernels.

This module is deliberately not empty. `config.py` reads `GOOGLE_CLOUD_PROJECT` with
`os.environ[...]`, strictly, at import time — a missing project must fail loudly rather
than silently talk to the wrong GCP project. That strictness means the `.env` file the
README tells you to create has to be loaded before *any* `kernelsmith.*` import runs,
and importing the package is the only hook that is guaranteed to happen first.

Nothing else in the repo loaded it: ADK reads `.env` only through the `adk` CLI, which
neither `make demo` nor `make serve-ui` goes through, so `cp .env.example .env` followed
by `make demo` died on a KeyError one line into startup.

`override=False` is the important argument. A real environment variable — the VM's
shell profile, a CI secret, `GOOGLE_CLOUD_PROJECT=... uv run ...` on the command line —
always wins over the file, so this can never silently redirect a run to another project.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _load_env_file() -> None:
    """Load repo-root `.env` into the environment without overriding what is already set.

    Falls back to a minimal parser if `python-dotenv` is unavailable, so an incomplete
    install still starts rather than failing on an import of a convenience library.
    """
    if not _ENV_FILE.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        for line in _ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        return
    load_dotenv(_ENV_FILE, override=False)


_load_env_file()
