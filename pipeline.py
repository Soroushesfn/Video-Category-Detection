"""Backward-compatible entry point used by the repository's CI workflow.

Full training requires the source database, which is intentionally not
distributed. When it is unavailable, this command performs a reproducibility
smoke test over the checked-in code and experiment artifacts instead.
"""

from __future__ import annotations

import argparse
import compileall
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DATABASE = ROOT / "Database" / "dataset.db"


def smoke_test() -> None:
    required_artifacts = [
        ROOT / "Models" / "youtube_category_classifier.keras",
        ROOT / "Embeddings" / "title_embeddings.npy",
        ROOT / "Embeddings" / "tags_embeddings.npy",
        ROOT / "Embeddings" / "description_embeddings.npy",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_artifacts if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing published artifacts: {', '.join(missing)}")
    if not compileall.compile_dir(ROOT / "Code", quiet=1):
        raise RuntimeError("Python source validation failed")
    print("[CHECK] Source database is not distributed; code and artifacts validated.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or validate the project pipeline.")
    parser.add_argument("--mode", choices=("train", "test"), required=True)
    args = parser.parse_args()

    if not SOURCE_DATABASE.exists():
        smoke_test()
        return

    mode = "train" if args.mode == "train" else "evaluate"
    subprocess.run(
        [sys.executable, str(ROOT / "Code" / "pipeline.py"), mode], check=True
    )


if __name__ == "__main__":
    main()
