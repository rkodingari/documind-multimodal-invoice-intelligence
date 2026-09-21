#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import requests

from documind.extraction.service import InvoiceExtractionService
from documind.schemas import primitive_result

CORE_FIELDS = [
    "invoice_number",
    "invoice_date",
    "vendor_name",
    "customer_name",
    "currency",
    "subtotal",
    "tax",
    "total",
]


def normalized(value: Any) -> str:
    if isinstance(value, float | int):
        return f"{float(value):.2f}"
    return " ".join(str(value or "").casefold().split())


def evaluate_provider(
    dataset_dir: Path, provider: str
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    records = json.loads((dataset_dir / "ground_truth.json").read_text())
    service = InvoiceExtractionService()
    exact_hits = normalized_hits = line_hits = line_total = document_hits = 0
    latency_total = cost_total = 0.0
    cost_observations = suspicious_tp = suspicious_fp = suspicious_fn = 0
    field_total = len(records) * len(CORE_FIELDS)
    failures: list[dict[str, Any]] = []
    for record in records:
        full_prediction = service.extract(
            dataset_dir / record["filename"], "application/pdf", provider
        )
        prediction = primitive_result(full_prediction)
        latency_total += full_prediction.metrics.latency_ms
        if full_prediction.metrics.estimated_cost_usd is not None:
            cost_total += full_prediction.metrics.estimated_cost_usd
            cost_observations += 1
        expected_suspicious = bool(record.get("suspicious", False))
        predicted_suspicious = full_prediction.review.required
        suspicious_tp += int(expected_suspicious and predicted_suspicious)
        suspicious_fp += int(not expected_suspicious and predicted_suspicious)
        suspicious_fn += int(expected_suspicious and not predicted_suspicious)
        truth = record["ground_truth"]
        exact = sum(prediction[field] == truth[field] for field in CORE_FIELDS)
        norm = sum(
            normalized(prediction[field]) == normalized(truth[field]) for field in CORE_FIELDS
        )
        exact_hits += exact
        normalized_hits += norm
        predicted_items = prediction["line_items"]
        truth_items = truth["line_items"]
        for index, truth_item in enumerate(truth_items):
            line_total += 1
            if index < len(predicted_items) and all(
                normalized(predicted_items[index].get(key)) == normalized(truth_item[key])
                for key in ("description", "quantity", "unit_price", "amount")
            ):
                line_hits += 1
        doc_ok = norm == len(CORE_FIELDS) and len(predicted_items) == len(truth_items)
        document_hits += int(doc_ok)
        if not doc_ok and len(failures) < 10:
            mismatches = {
                field: {"expected": truth[field], "predicted": prediction[field]}
                for field in CORE_FIELDS
                if normalized(prediction[field]) != normalized(truth[field])
            }
            failures.append(
                {
                    "filename": record["filename"],
                    "input_mode": "ocr" if record["scanned"] else "text",
                    "field_mismatches": mismatches,
                    "expected_line_items": len(truth_items),
                    "predicted_line_items": len(predicted_items),
                    "likely_cause": "OCR noise or layout ambiguity"
                    if record["scanned"]
                    else "Rule/layout mismatch",
                }
            )
    metrics = {
        "documents": len(records),
        "exact_match": exact_hits / field_total if field_total else 0,
        "normalized_field_accuracy": normalized_hits / field_total if field_total else 0,
        "line_item_accuracy": line_hits / line_total if line_total else 0,
        "document_accuracy": document_hits / len(records) if records else 0,
        "average_latency_ms": latency_total / len(records) if records else 0,
        "total_cost_usd": cost_total if cost_observations else None,
        "suspicion_precision": suspicious_tp / (suspicious_tp + suspicious_fp)
        if suspicious_tp + suspicious_fp
        else 0,
        "suspicion_recall": suspicious_tp / (suspicious_tp + suspicious_fn)
        if suspicious_tp + suspicious_fn
        else 0,
    }
    return metrics, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DocuMind extraction on labeled invoices")
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/generated"))
    parser.add_argument(
        "--provider",
        choices=[
            "rules",
            "learned",
            "openai",
            "ollama",
            "pretrained_vlm",
            "finetuned_layoutlm",
            "all",
        ],
        default="rules",
    )
    args = parser.parse_args()
    providers = (
        [
            "rules",
            "learned",
            "openai",
            "ollama",
            "pretrained_vlm",
            "finetuned_layoutlm",
        ]
        if args.provider == "all"
        else [args.provider]
    )
    print(
        "| Provider | N | Exact | Normalized | Lines | Documents | Latency | Cost | Suspicion P/R |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    all_failures = {}
    for provider in providers:
        try:
            metrics, failures = evaluate_provider(args.dataset, provider)
        except (RuntimeError, ValueError, requests.RequestException) as exc:
            print(f"| {provider} | — | — | — | — | — | — | — | unavailable: {exc} |")
            all_failures[provider] = {"status": "not_run", "reason": str(exc)}
            continue
        all_failures[provider] = failures
        print(
            f"| {provider} | {int(metrics['documents'])} | "
            f"{metrics['exact_match']:.1%} | "
            f"{metrics['normalized_field_accuracy']:.1%} | {metrics['line_item_accuracy']:.1%} | "
            f"{metrics['document_accuracy']:.1%} | {metrics['average_latency_ms']:.1f} ms | "
            f"{metrics['total_cost_usd'] if metrics['total_cost_usd'] is not None else 'n/a'} | "
            f"{metrics['suspicion_precision']:.1%}/{metrics['suspicion_recall']:.1%} |"
        )
    failure_path = args.dataset / "failure_examples.json"
    failure_path.write_text(json.dumps(all_failures, indent=2))
    print(f"\nFailure examples written to {failure_path}")


if __name__ == "__main__":
    main()
