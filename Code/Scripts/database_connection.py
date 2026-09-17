"""SQLite connection helper."""

import sqlite3
from pathlib import Path


def get_db_connection(database_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection and surface a clear error on failure."""
    try:
        return sqlite3.connect(database_path)
    except sqlite3.Error as error:
        raise ConnectionError(
            f"Could not connect to SQLite database at {database_path}"
        ) from error
