"""Static configuration and environment wiring."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Repo root: src/kickbase_xp/config.py -> up three levels.
ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DB_PATH = ROOT / "data" / "history.sqlite"
# Committed, append-only daily archive -- see archive.py for why the
# SQLite file itself is not the thing under version control.
DEFAULT_SNAPSHOT_DIR = ROOT / "data" / "snapshots"
DEFAULT_OUT_DIR = ROOT / "site"

API_BASE = "https://api.kickbase.com"

# Kickbase competition ids. v1 predicts Bundesliga only, but nothing below is
# hard-wired to it -- the competition id is threaded through as a parameter.
COMPETITION_BUNDESLIGA = "1"

# The performance feed labels each season with its competition name rather
# than an id, and mixes competitions into one list -- so this is what the
# feature builder filters on. Keys are Kickbase competition ids.
COMPETITION_NAMES = {
    "1": "Bundesliga",
    "2": "2. Bundesliga",
}
DEFAULT_COMPETITION_NAME = COMPETITION_NAMES[COMPETITION_BUNDESLIGA]

# Kickbase position codes.
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Kickbase player status codes. 0 is the only "available" state; everything
# else feeds the hard override on P(plays) (see model.STATUS_PLAY_CAP).
STATUS_FIT = 0
STATUS_LABELS = {
    0: "fit",
    1: "injured",
    2: "questionable",
    4: "suspended",
    8: "absent",
    16: "rehab",
    32: "unknown",
}

# Per-matchday `st` in the performance feed: 5 == took part.
PERF_STATUS_PLAYED = 5

# Kickbase market-value history is expressed in days since the Unix epoch.
MV_EPOCH_DAYS = "1970-01-01"


@dataclass(frozen=True)
class Credentials:
    email: str
    password: str


def load_credentials() -> Credentials:
    """Read Kickbase credentials from the environment.

    `KICK_EMAIL`/`KICK_PASS` are the documented names; `KICK_USER` is accepted
    as an alias because the project plan named it that way.
    """
    email = os.environ.get("KICK_EMAIL") or os.environ.get("KICK_USER")
    password = os.environ.get("KICK_PASS") or os.environ.get("KICK_PASSWORD")
    if not email or not password:
        raise SystemExit(
            "Missing Kickbase credentials. Set KICK_EMAIL and KICK_PASS "
            "(a .env file in the repo root is picked up automatically)."
        )
    return Credentials(email=email, password=password)


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader so local runs match the Action's environment."""
    path = path or (ROOT / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
