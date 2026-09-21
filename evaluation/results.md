# Evaluation results

Measured locally on 2026-06-29 using a 180-document training corpus (seed 17) and a separate
30-document text-native evaluation corpus (seed 1337). Three evaluation invoices contain deliberate
8% arithmetic inconsistencies. Commands:

```bash
python scripts/generate_invoices.py --count 180 --scan-every 0 --seed 17 \
  --output training/generated
python scripts/generate_invoices.py --count 30 --scan-every 0 --seed 1337 \
  --anomaly-every 10 --output evaluation/generated
python scripts/train_ranker.py
python evaluation/evaluate.py --provider rules
python evaluation/evaluate.py --provider learned
```

## Candidate-ranking model

| Candidate metric | Result |
|---|---:|
| Accuracy | 98.60% |
| Precision | 99.75% |
| Recall | 96.37% |
| F1 | 98.03% |
| Brier score | 0.0091 |
| Training candidates | 6,770 |
| Validation candidates | 1,144 |

## End-to-end extraction and routing

| Provider | N | Exact | Normalized | Lines | Documents | Avg latency | Cost | Suspicion P/R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rules | 30 | 100.0% | 100.0% | 100.0% | 100.0% | 25.7 ms | n/a | 10.0% / 100.0% |
| learned | 30 | 100.0% | 100.0% | 100.0% | 100.0% | 48.5 ms | n/a | 100.0% / 100.0% |
| OpenAI | not run | — | — | — | — | — | — | — |
| Ollama | not run | — | — | — | — | — | — | — |
| pretrained VLM | not run | — | — | — | — | — | — | — |
| fine-tuned LayoutLMv3 | not run | — | — | — | — | — | — | — |

Interpretation: the rules baseline extracted the controlled layouts correctly but its fixed vendor
confidence routed every document to review. The calibrated learned probabilities eliminated those
false-positive review routes while arithmetic validation still caught all three inconsistent totals.

These are in-distribution synthetic measurements, not a claim about production invoices. The local
host did not have Tesseract, a GPU, Ollama, or a paid API credential, so results for those execution
paths are deliberately marked `not run`. CI installs Tesseract and exercises the mixed text/scan path.
