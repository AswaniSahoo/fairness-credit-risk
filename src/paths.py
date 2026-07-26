"""Filesystem locations derived from the repository layout.

Every path in the project resolves from PROJECT_ROOT so the code runs unchanged on any
machine. Replaces the absolute ``/home/aswani/automl`` paths that previously made the
project unrunnable outside their author's machine.
"""

from pathlib import Path

# paths.py lives at <root>/src/paths.py, so the root is two parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"


def ensure_output_dirs() -> None:
    """Create the artifact and report directories if they do not yet exist."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
