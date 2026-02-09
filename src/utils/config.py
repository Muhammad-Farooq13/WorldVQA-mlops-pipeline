"""
Configuration helpers for the WorldVQA project.

Centralizes common paths and constants to avoid hard-coding them
all over the codebase.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def ensure_directories() -> None:
    """
    Ensure that all key directories exist.
    """
    for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

