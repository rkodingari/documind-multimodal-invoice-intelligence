# Baseline failure examples

These cases were executed directly against `extract_with_rules` on 2026-06-29. They illustrate
known limits without inventing benchmark results.

| Case | Expected | Observed | Cause |
|---|---|---|---|
| `Date: 03/04/2025` intended as 3 April | `2025-04-03` | `2025-03-04` | A numeric date without locale is inherently ambiguous; the baseline tries month-first. |
| Totals have no symbol or ISO code | a known currency | `null` | Currency cannot be inferred safely from numbers alone. |
| Description wraps before `services \| 2 \| $50.00 \| $100.00` | `Professional consulting and implementation services` | `services` | The row regex operates line by line and cannot join wrapped descriptions. |

Likely OCR-specific failures include `O/0` confusion in invoice identifiers, decimal separators lost
in low-resolution scans, and columns merged by OCR. The evaluation script writes concrete mismatches
to `evaluation/generated/failure_examples.json` whenever they occur.

