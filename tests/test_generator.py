import random

from scripts.generate_invoices import build_invoice


def test_synthetic_ground_truth_is_arithmetically_consistent():
    invoice = build_invoice(0, random.Random(42))
    assert sum(item["amount"] for item in invoice["line_items"]) == invoice["subtotal"]
    assert round(invoice["subtotal"] + invoice["tax"], 2) == invoice["total"]
