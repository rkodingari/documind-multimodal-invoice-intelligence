from documind.schemas import InvoicePrediction, ValidationResult


def validate_prediction(
    prediction: InvoicePrediction, tolerance: float = 0.02
) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    subtotal = prediction.subtotal.value
    tax = prediction.tax.value
    total = prediction.total.value
    if all(isinstance(value, int | float) for value in (subtotal, tax, total)):
        expected = round(float(subtotal) + float(tax), 2)
        difference = round(abs(expected - float(total)), 2)
        allowed = max(0.02, abs(float(total)) * tolerance)
        passed = difference <= allowed
        results.append(
            ValidationResult(
                rule="subtotal_plus_tax_equals_total",
                passed=passed,
                message=(
                    "Subtotal plus tax agrees with total."
                    if passed
                    else f"Expected {expected:.2f}, but extracted total is {float(total):.2f}."
                ),
                expected=expected,
                actual=float(total),
                difference=difference,
            )
        )
    else:
        results.append(
            ValidationResult(
                rule="subtotal_plus_tax_equals_total",
                passed=False,
                message="Cannot validate arithmetic because one or more totals are missing.",
            )
        )

    if prediction.line_items and isinstance(subtotal, int | float):
        item_sum = round(sum(item.amount for item in prediction.line_items), 2)
        difference = round(abs(item_sum - float(subtotal)), 2)
        passed = difference <= max(0.02, abs(float(subtotal)) * tolerance)
        results.append(
            ValidationResult(
                rule="line_items_equal_subtotal",
                passed=passed,
                message=(
                    "Line-item amounts agree with subtotal."
                    if passed
                    else f"Line items sum to {item_sum:.2f}; subtotal is {float(subtotal):.2f}."
                ),
                expected=item_sum,
                actual=float(subtotal),
                difference=difference,
            )
        )
    return results


def calculate_overall_confidence(prediction: InvoicePrediction) -> float:
    fields = [
        prediction.invoice_number,
        prediction.invoice_date,
        prediction.vendor_name,
        prediction.customer_name,
        prediction.currency,
        prediction.subtotal,
        prediction.tax,
        prediction.total,
    ]
    scores = [field.confidence for field in fields]
    scores.extend(item.confidence for item in prediction.line_items)
    if not scores:
        return 0.0
    score = sum(scores) / len(scores)
    if any(not result.passed for result in prediction.validations):
        score *= 0.85
    return round(min(1.0, max(0.0, score)), 3)
