from pathlib import Path

import fitz

from documind.extraction import document as document_module


def _save_pdf(path: Path, text: str | None = None) -> None:
    pdf = fitz.open()
    page = pdf.new_page()
    if text:
        page.insert_text((50, 80), text)
    pdf.save(path)
    pdf.close()


def test_pdf_with_embedded_text_uses_text_path(tmp_path):
    path = tmp_path / "text.pdf"
    content = "Invoice No: TEST-1\n" + "A sufficiently long embedded invoice text. " * 3
    _save_pdf(path, content)

    extracted = document_module.extract_document_text(path, "application/pdf")

    assert extracted.method == "text"
    assert "TEST-1" in extracted.text


def test_image_only_pdf_uses_ocr_path(tmp_path, monkeypatch):
    path = tmp_path / "scan.pdf"
    _save_pdf(path)
    monkeypatch.setattr(
        document_module,
        "_ocr_image_with_tokens",
        lambda _image, _page, _width, _height: ("OCR invoice text", []),
    )

    extracted = document_module.extract_document_text(path, "application/pdf")

    assert extracted.method == "ocr"
    assert extracted.text == "OCR invoice text"
