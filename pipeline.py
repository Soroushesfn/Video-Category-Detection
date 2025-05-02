import subprocess

subprocess.run(["python", "scripts/load_data.py"])
subprocess.run(["python", "scripts/embed_text_columns.py"])
subprocess.run(["python", "scripts/feature_engineering.py"])
subprocess.run(["python", "scripts/preprocess.py"])
