#!/usr/bin/env python3
"""Train and benchmark LightGBM on the Kaggle credit-card fraud dataset."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark LightGBM fraud detection on creditcard.csv."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("creditcard.csv"),
        help="Path to creditcard.csv (default: ./creditcard.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_result.json"),
        help="Result JSON path (default: ./benchmark_result.json)",
    )
    return parser.parse_args()


def timed_predict(model: LGBMClassifier, features: pd.DataFrame) -> float:
    start = time.perf_counter()
    model.predict_proba(features, num_iteration=model.best_iteration_)
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Download creditcard.csv first."
        )

    load_start = time.perf_counter()
    data = pd.read_csv(data_path)
    load_seconds = time.perf_counter() - load_start

    required_columns = {"Class"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    features = data.drop(columns="Class")
    target = data["Class"].astype(int)

    # Keep the test set untouched. A validation split is used only for early stopping.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    positive_count = int((y_train == 1).sum())
    if positive_count == 0:
        raise ValueError("Training split contains no fraud samples.")

    model = LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.80,
        colsample_bytree=0.80,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    training_start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_X=X_valid,
        eval_y=y_valid,
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    training_seconds = time.perf_counter() - training_start

    validation_probabilities = model.predict_proba(
        X_valid, num_iteration=model.best_iteration_
    )[:, 1]
    validation_precision, validation_recall, thresholds = precision_recall_curve(
        y_valid, validation_probabilities
    )
    threshold_index = max(
        range(len(thresholds)),
        key=lambda index: (
            2
            * validation_precision[index]
            * validation_recall[index]
            / (validation_precision[index] + validation_recall[index] + 1e-15)
        ),
    )
    decision_threshold = float(thresholds[threshold_index])

    probabilities = model.predict_proba(
        X_test, num_iteration=model.best_iteration_
    )[:, 1]
    predictions = (probabilities >= decision_threshold).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
    }

    single_row = X_test.iloc[[0]]
    batch_1000 = X_test.iloc[:1000]

    # Warm up code paths before collecting inference timings.
    model.predict_proba(single_row, num_iteration=model.best_iteration_)
    model.predict_proba(batch_1000, num_iteration=model.best_iteration_)

    single_row_times = [timed_predict(model, single_row) for _ in range(100)]
    batch_times = [timed_predict(model, batch_1000) for _ in range(20)]
    median_single_seconds = statistics.median(single_row_times)
    median_batch_seconds = statistics.median(batch_times)

    result = {
        "dataset": {
            "path": str(data_path),
            "rows": int(len(data)),
            "features": int(features.shape[1]),
            "fraud_rows": int(target.sum()),
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_valid)),
            "test_rows": int(len(X_test)),
        },
        "timings_seconds": {
            "data_load": float(load_seconds),
            "training": float(training_seconds),
        },
        "model": {
            "best_iteration": int(model.best_iteration_),
            "class_weight": "balanced",
            "decision_threshold": decision_threshold,
            "threshold_selection": "maximum F1 on validation set",
        },
        "metrics": metrics,
        "inference": {
            "latency_1_row_ms": float(median_single_seconds * 1000),
            "latency_repetitions": len(single_row_times),
            "batch_1000_duration_ms": float(median_batch_seconds * 1000),
            "throughput_1000_rows_per_second": float(
                len(batch_1000) / median_batch_seconds
            ),
            "throughput_repetitions": len(batch_times),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rows = [
        ("Thoi gian load data", f"{load_seconds:.4f} s"),
        ("Thoi gian training", f"{training_seconds:.4f} s"),
        ("Best iteration", str(model.best_iteration_)),
        ("AUC-ROC", f"{metrics['auc_roc']:.6f}"),
        ("Accuracy", f"{metrics['accuracy']:.6f}"),
        ("F1-Score", f"{metrics['f1_score']:.6f}"),
        ("Precision", f"{metrics['precision']:.6f}"),
        ("Recall", f"{metrics['recall']:.6f}"),
        ("Inference latency (1 row)", f"{median_single_seconds * 1000:.4f} ms"),
        (
            "Inference throughput (1000 rows)",
            f"{len(batch_1000) / median_batch_seconds:.2f} rows/s",
        ),
    ]

    print("\n| Metric | Ket qua |")
    print("|---|---:|")
    for name, value in rows:
        print(f"| {name} | {value} |")
    print(f"\nSaved full results to: {output_path}")


if __name__ == "__main__":
    main()
