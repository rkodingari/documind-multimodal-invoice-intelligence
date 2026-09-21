import fitz
from fastapi.testclient import TestClient

from documind.main import app


def make_invoice_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    lines = [
        "Northstar Analytics",
        "INVOICE",
        "Invoice No: API-1001",
        "Invoice Date: 2025-02-14",
        "Bill To: Acme Retail Ltd",
        "Consulting package | 2 | $100.00 | $200.00",
        "Subtotal: $200.00",
        "Tax: $20.00",
        "Total: $220.00",
    ]
    y = 60
    for line in lines:
        page.insert_text((50, y), line, fontsize=11)
        y += 25
    payload = doc.tobytes()
    doc.close()
    return payload


def test_upload_correct_and_export_round_trip():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200

        response = client.post(
            "/api/v1/documents",
            files={"file": ("invoice.pdf", make_invoice_pdf(), "application/pdf")},
        )
        assert response.status_code == 201, response.text
        document = response.json()
        assert document["active_result"]["invoice_number"]["value"] == "API-1001"

        correction = client.put(
            f"/api/v1/documents/{document['id']}/corrections",
            json={"customer_name": "Corrected Customer"},
        )
        assert correction.status_code == 200
        corrected = correction.json()
        assert corrected["prediction"]["customer_name"]["value"] == "Acme Retail Ltd"
        assert corrected["corrected"]["customer_name"]["value"] == "Corrected Customer"

        exported = client.get(f"/api/v1/documents/{document['id']}/export?format=json")
        assert exported.status_code == 200
        assert exported.json()["customer_name"] == "Corrected Customer"

        exported_csv = client.get(f"/api/v1/documents/{document['id']}/export?format=csv")
        assert exported_csv.status_code == 200
        assert "Corrected Customer" in exported_csv.text

        original_file = client.get(f"/api/v1/documents/{document['id']}/file")
        assert original_file.status_code == 200
        assert original_file.headers["content-type"].startswith("application/pdf")

        evidence = client.get(f"/api/v1/documents/{document['id']}/evidence/1")
        assert evidence.status_code == 200
        assert evidence.headers["content-type"] == "image/png"

        providers = client.get("/api/v1/providers")
        assert providers.status_code == 200
        assert {item["name"] for item in providers.json()} >= {"rules", "learned", "openai"}


def test_rejects_unsupported_file():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/documents",
            files={"file": ("notes.txt", b"not an invoice", "text/plain")},
        )
        assert response.status_code == 415


def test_learned_provider_returns_metrics_and_evidence():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/documents?provider=learned",
            files={"file": ("learned-invoice.pdf", make_invoice_pdf(), "application/pdf")},
        )

        assert response.status_code == 201, response.text
        result = response.json()["active_result"]
        assert result["provider"] == "learned"
        assert result["invoice_number"]["source"] == "learned"
        assert result["invoice_number"]["evidence"]
        assert result["metrics"]["latency_ms"] > 0
