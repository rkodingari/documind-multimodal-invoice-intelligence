from pathlib import Path

import fitz

from documind.extraction.document import extract_document_text
from documind.extraction.providers import (
    ModelField,
    ModelInvoice,
    RulesProvider,
    prediction_from_model,
)
from documind.extraction.risk import assess_risk
from documind.ml.ranker import generate_candidates
from scripts.finetune_layoutlm import matching_indices


def make_layout_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    lines = [
        "Vendor: Atlas Systems",
        "Invoice No: ML-2048",
        "Invoice Date: 2025-04-10",
        "Bill To: Example Buyer",
        "Subtotal: $100.00",
        "Tax: $10.00",
        "Total: $110.00",
    ]
    for index, line in enumerate(lines):
        page.insert_text((50, 60 + index * 28), line)
    document.save(path)
    document.close()


def model_field(value, page=1, bbox=None, text=None, confidence=0.9):
    return ModelField(
        value=value,
        confidence=confidence,
        page=page,
        bbox=bbox or [50, 50, 250, 80],
        supporting_text=text or str(value),
    )


def test_rules_attach_page_region_evidence_and_candidates(tmp_path):
    path = tmp_path / "layout.pdf"
    make_layout_pdf(path)
    artifact = extract_document_text(path, "application/pdf")

    prediction = RulesProvider().extract(artifact)
    candidates = generate_candidates(artifact, "invoice_number")

    assert prediction.invoice_number.evidence[0].page == 1
    assert prediction.invoice_number.evidence[0].bbox[2] > 50
    assert any(candidate.value == "ML-2048" for candidate in candidates)
    assert candidates[0].features["relative_y"] >= 0
    assert matching_indices(artifact.pages[0].tokens, "invoice_number", "ML-2048")
    assert matching_indices(artifact.pages[0].tokens, "subtotal", 100.0)


def test_model_output_is_normalized_and_evidence_is_validated(tmp_path):
    path = tmp_path / "layout.pdf"
    make_layout_pdf(path)
    artifact = extract_document_text(path, "application/pdf")
    payload = ModelInvoice(
        invoice_number=model_field("ML-2048"),
        invoice_date=model_field("April 10, 2025"),
        vendor_name=model_field("Atlas Systems"),
        customer_name=model_field("Example Buyer"),
        currency=model_field("USD"),
        subtotal=model_field("$100.00"),
        tax=model_field("$10.00"),
        total=model_field("$110.00"),
        line_items=[],
    )

    prediction = prediction_from_model(payload, artifact, "test_vlm")

    assert prediction.invoice_date.value == "2025-04-10"
    assert prediction.total.value == 110.0
    assert prediction.total.evidence[0].page == 1
    assert prediction.provider == "test_vlm"


def test_suspicion_and_human_review_routing():
    from documind.extraction.rules import extract_with_rules

    prediction = extract_with_rules(
        "Vendor: Atlas\nInvoice No: X-1\nDate: 2025-01-01\nBill To: Buyer\n"
        "Subtotal: $100\nTax: $75\nTotal: $120"
    )
    assess_risk(prediction)

    codes = {signal.code for signal in prediction.suspicion_signals}
    assert "unusually_high_tax" in codes
    assert any(code.startswith("validation_failed") for code in codes)
    assert prediction.review.required
