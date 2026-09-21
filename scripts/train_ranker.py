#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from documind.extraction.document import extract_document_text
from documind.ml.ranker import TARGET_FIELDS, generate_candidates


def normalized(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.2f}"
    return " ".join(str(value or "").casefold().split())


def build_rows(dataset: Path) -> tuple[list[dict[str, Any]], list[int], list[str]]:
    records = json.loads((dataset / "ground_truth.json").read_text())
    features: list[dict[str, Any]] = []
    labels: list[int] = []
    groups: list[str] = []
    for record in records:
        artifact = extract_document_text(dataset / record["filename"], "application/pdf")
        truth = record["ground_truth"]
        for field in TARGET_FIELDS:
            candidates = generate_candidates(artifact, field)
            for candidate in candidates:
                features.append(candidate.features)
                labels.append(int(normalized(candidate.value) == normalized(truth[field])))
                groups.append(record["filename"])
    return features, labels, groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DocuMind's local candidate ranker")
    parser.add_argument("--train", type=Path, default=Path("training/generated"))
    parser.add_argument("--validation", type=Path, default=Path("evaluation/generated"))
    parser.add_argument("--output", type=Path, default=Path("models/candidate_ranker.joblib"))
    args = parser.parse_args()
    train_x, train_y, _ = build_rows(args.train)
    valid_x, valid_y, _ = build_rows(args.validation)
    if not train_x or len(set(train_y)) < 2:
        raise SystemExit("Training data must contain positive and negative field candidates")
    pipeline = Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            (
                "classifier",
                CalibratedClassifierCV(
                    LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
                    cv=3,
                    method="sigmoid",
                ),
            ),
        ]
    )
    pipeline.fit(train_x, train_y)
    probabilities = pipeline.predict_proba(valid_x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_y, predictions, average="binary", zero_division=0
    )
    metrics = {
        "candidate_accuracy": accuracy_score(valid_y, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": brier_score_loss(valid_y, probabilities),
        "train_candidates": len(train_y),
        "validation_candidates": len(valid_y),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "metadata": {
                "version": 1,
                "target_fields": list(TARGET_FIELDS),
                "validation_metrics": metrics,
            },
        },
        args.output,
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
