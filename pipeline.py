import subprocess
import argparse

parser = argparse.ArgumentParser(description="Run data pipeline")
parser.add_argument(
    "--mode",
    type=str,
    choices=["train", "test"],
    help="Mode to run the pipeline: 'train' or 'test'"
)
args = parser.parse_args()

if args.mode == "train":
    subprocess.run(["python", "scripts/load_data.py"])
    subprocess.run(["python", "scripts/feature_engineering.py"])
    subprocess.run(["python", "scripts/preprocess.py"])
    subprocess.run(["python", "scripts/splitData.py"])
    subprocess.run(["python", "scripts/train_model.py"])

elif args.mode == "test":
    subprocess.run(["python", "scripts/make_prediction.py"])

else:
    print("No valid mode provided — doing nothing.")
