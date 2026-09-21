# ML methodology

## CPU candidate ranker

The local learned extractor is a supervised ranking system. Deterministic parsing generates possible
values for each field; a calibrated logistic classifier estimates whether each candidate is correct.
The highest-probability candidate becomes the field prediction.

Features include:

- Target field identity and nearby label cues.
- Date, numeric, currency, and text shape.
- Relative x/y page position and line position.
- OCR confidence and page index.
- Conflicting cues such as `subtotal`, `tax`, and `total` on the same line.

Training uses document-level train/evaluation separation. Candidate accuracy and Brier score measure
classification and confidence calibration; end-to-end field metrics measure whether ranking chooses
the correct value. The model does not learn line-item tables yet—rules remain the fallback there.

## LayoutLMv3

The GPU fine-tuning path is token classification over three modalities:

- Document image pixels.
- OCR/PDF word tokens.
- Normalized two-dimensional bounding boxes.

Ground-truth values are aligned back to evidence regions and converted to BIO labels. This makes the
task genuinely layout-aware instead of treating OCR output as plain text. The training script is
functional but no checkpoint metric is claimed until it is actually trained on GPU hardware.

## VLM providers

OpenAI, Ollama, and Hugging Face providers receive rendered page images and a shared JSON schema.
They must cite the page, original-coordinate bounding box, and supporting text for every value.
Application-side Pydantic validation, normalization, arithmetic rules, and human review apply even
when the model produces schema-constrained output.

## Suspicion and review routing

Suspicion is intentionally separate from extraction. Signals include arithmetic inconsistency,
line-item/subtotal mismatch, missing fields, non-positive totals, unusually high tax, future dates,
duplicate invoice numbers, and low calibrated confidence. Medium/high signals or low-confidence
required fields route a document to human review.

