"""Command-line entry point for the video-category pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent / "Scripts"
TRAIN_STAGES = (
    "load_data.py",
    "embed_text_columns.py",
    "feature_engineering.py",
    "preprocess.py",
    "split_data.py",
    "train_model.py",
)
EVALUATION_STAGES = ("make_prediction.py", "predictions.py")


def run_stage(script_name: str) -> None:
    """Run one stage with the active Python interpreter."""
    script_path = SCRIPT_DIR / script_name
    print(f"[PIPELINE] Running {script_name}", flush=True)
    subprocess.run([sys.executable, str(script_path)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or evaluate the YouTube video-category classifier."
    )
    parser.add_argument("mode", choices=("train", "evaluate"))
    args = parser.parse_args()

    stages = TRAIN_STAGES if args.mode == "train" else EVALUATION_STAGES
    for stage in stages:
        run_stage(stage)


if __name__ == "__main__":
    main()
