from pathlib import Path

import fitz

from documind.extraction.document import extract_document_text
from documind.extraction.providers import LearnedProvider


def test_trained_ranker_selects_fields_and_probabilities(tmp_path):
    pdf_path = tmp_path / "learned.pdf"
    document = fitz.open()
    page = document.new_page()
    for index, line in enumerate(
        [
            "Atlas Systems",
            "Invoice No: ML-999",
            "Invoice Date: 2025-05-01",
            "Bill To: Example Buyer",
            "Subtotal: $100.00",
            "Tax: $10.00",
            "Total: $110.00",
        ]
    ):
        page.insert_text((50, 60 + index * 25), line)
    document.save(pdf_path)
    document.close()
    artifact = extract_document_text(pdf_path, "application/pdf")

    result = LearnedProvider(Path("models/candidate_ranker.joblib")).extract(artifact)

    assert result.invoice_number.value == "ML-999"
    assert result.invoice_number.source == "learned"
    assert 0 < result.invoice_number.confidence <= 1
    assert result.invoice_number.evidence
    assert result.provider == "learned"
