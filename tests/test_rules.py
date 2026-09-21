from documind.extraction.rules import extract_with_rules

INVOICE_TEXT = """
Northstar Analytics
INVOICE
Invoice No: DM-2025-1001
Invoice Date: 2025-01-08
Bill To: Acme Retail Ltd

Description | Qty | Unit Price | Amount
Data analysis workshop | 2 | $450.00 | $900.00
Cloud support hours | 3 | $95.00 | $285.00

Subtotal: $1,185.00
Tax: $118.50
Total: $1,303.50
"""


def test_rule_extraction_happy_path():
    result = extract_with_rules(INVOICE_TEXT)
    assert result.invoice_number.value == "DM-2025-1001"
    assert result.invoice_date.value == "2025-01-08"
    assert result.vendor_name.value == "Northstar Analytics"
    assert result.customer_name.value == "Acme Retail Ltd"
    assert result.currency.value == "USD"
    assert result.subtotal.value == 1185.0
    assert result.tax.value == 118.5
    assert result.total.value == 1303.5
    assert len(result.line_items) == 2
    assert all(check.passed for check in result.validations)


def test_arithmetic_mismatch_is_flagged():
    result = extract_with_rules(INVOICE_TEXT.replace("$1,303.50", "$1,400.00"))
    validation = next(v for v in result.validations if v.rule == "subtotal_plus_tax_equals_total")
    assert not validation.passed
    assert result.overall_confidence < 0.9


def test_ocr_confidence_is_lower_than_text():
    text_result = extract_with_rules(INVOICE_TEXT, "text")
    ocr_result = extract_with_rules(INVOICE_TEXT, "ocr")
    assert ocr_result.overall_confidence < text_result.overall_confidence
