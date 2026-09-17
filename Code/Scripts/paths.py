"""Shared filesystem paths for pipeline stages."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_DIR = PROJECT_ROOT / "Database"
MODEL_DIR = PROJECT_ROOT / "Models"
TMP_DIR = PROJECT_ROOT / "tmp"
EMBEDDING_DIR = TMP_DIR / "embeddings"


def create_runtime_directories() -> None:
    """Create directories used for generated artifacts."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
