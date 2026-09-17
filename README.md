# YouTube Video Category Classification

An end-to-end machine-learning pipeline for classifying trending YouTube videos from semantic text representations and engagement metadata. Developed for the University of Tehran Data Science course.

## Results

The final multi-branch neural network achieved the following held-out performance:

| Metric | Score |
| --- | ---: |
| Accuracy | **78.91%** |
| Precision | **78.98%** |
| Recall | **75.78%** |

It outperformed the project's classical machine-learning and single-stream MLP baselines. The metrics above are preserved from the final experimental report.

## Approach

The pipeline combines four complementary feature branches:

1. **Title semantics:** 384-dimensional MiniLM sentence embeddings.
2. **Tag semantics:** 384-dimensional embeddings of normalized video tags.
3. **Description semantics:** 384-dimensional description embeddings.
4. **Engagement metadata:** views, comments, engagement rate, like/dislike ratio, and tag count.

Each text representation passes through its own dense branch. A fourth branch processes numeric features; the four representations are then fused for multiclass prediction. The repository includes the trained Keras checkpoint and the three generated embedding matrices.

## Pipeline

```text
SQLite source data
  -> deduplication
  -> MiniLM text embeddings
  -> engagement feature engineering
  -> normalization and stratified splitting
  -> multi-branch MLP training
  -> evaluation and SQLite prediction export
```

Run the workflow from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python Code/pipeline.py train
python Code/pipeline.py evaluate
```

The training workflow expects `Database/dataset.db` with a table named `US_Trending_Videos`. The source dataset is not redistributed in this repository. If starting from the corresponding CSV export, place it at `Database/Data.csv` and run:

```bash
python Code/Scripts/import_to_db.py
```

Generated intermediate files are written to `tmp/` and excluded from version control.

## Repository structure

| Path | Contents |
| --- | --- |
| `Code/pipeline.py` | Train/evaluate command-line entry point |
| `Code/Scripts/` | Modular data, feature, training, and evaluation stages |
| `Models/` | Trained Keras classifier |
| `Embeddings/` | Saved title, tag, and description embeddings from the reported experiment |
| `Database/queries/` | SQL analyses and result visualizations |
| `Reports/` | Exploratory analysis notebook |
| `Project Presentation.pptx` | Final experimental report |

## Reproducibility notes

- Stratified data partitions use a fixed random seed of `44`.
- MLflow records training parameters, validation metrics, and the resulting model.
- The checked-in model and embeddings support inspection of the reported experiment without rerunning embedding generation.
- Full retraining requires the original source data and can be computationally intensive.
