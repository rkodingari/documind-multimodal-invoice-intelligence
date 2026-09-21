import base64
import json
import os
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("DOCUMIND_API_URL", "http://localhost:8000").rstrip("/")
FIELDS = [
    ("invoice_number", "Invoice number"),
    ("invoice_date", "Invoice date"),
    ("vendor_name", "Vendor"),
    ("customer_name", "Customer"),
    ("currency", "Currency"),
    ("subtotal", "Subtotal"),
    ("tax", "Tax"),
    ("total", "Total"),
]

st.set_page_config(page_title="DocuMind", page_icon="🧾", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1500px;}
    [data-testid="stMetricValue"] {font-size: 1.35rem;}
    .confidence-low {color: #b42318; font-weight: 700;}
    .confidence-ok {color: #067647; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)


def api_request(method: str, path: str, **kwargs: Any) -> requests.Response:
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=120, **kwargs)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        detail = getattr(exc.response, "text", str(exc))
        st.error(f"Backend request failed: {detail}")
        st.stop()


def render_document(document: dict[str, Any]) -> None:
    response = api_request("GET", f"/api/v1/documents/{document['id']}/file")
    if document["content_type"] == "application/pdf":
        encoded = base64.b64encode(response.content).decode()
        iframe = (
            f'<iframe src="data:application/pdf;base64,{encoded}" '
            'width="100%" height="780"></iframe>'
        )
        st.markdown(iframe, unsafe_allow_html=True)
    else:
        st.image(response.content, caption=document["filename"], use_container_width=True)


def load_providers() -> list[dict[str, Any]]:
    try:
        response = requests.get(f"{API_URL}/api/v1/providers", timeout=2)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return [{"name": "rules", "available": True, "reason": None, "requires": []}]


def render_evidence(document: dict[str, Any], result: dict[str, Any]) -> None:
    pages = {
        evidence["page"] for name, _label in FIELDS for evidence in result[name].get("evidence", [])
    }
    pages.update(
        evidence["page"] for item in result["line_items"] for evidence in item.get("evidence", [])
    )
    page_number = st.selectbox("Evidence page", sorted(pages or {1}))
    response = api_request("GET", f"/api/v1/documents/{document['id']}/evidence/{page_number}")
    st.image(
        response.content, caption=f"Cited regions · page {page_number}", use_container_width=True
    )


st.title("DocuMind")
st.caption("Multimodal invoice extraction with confidence-aware human review")

with st.sidebar:
    st.header("New invoice")
    uploaded = st.file_uploader("PDF, PNG, or JPEG", type=["pdf", "png", "jpg", "jpeg"])
    capabilities = load_providers()
    provider_names = [item["name"] for item in capabilities]
    provider = st.selectbox(
        "Extraction provider",
        provider_names,
        help="Rules and learned ML run locally; other providers are optional.",
    )
    selected_capability = next(item for item in capabilities if item["name"] == provider)
    if not selected_capability["available"]:
        st.warning(selected_capability["reason"] or "Provider is not currently available.")
    if st.button(
        "Extract invoice",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None or not selected_capability["available"],
    ):
        with st.spinner("Reading and extracting invoice…"):
            response = api_request(
                "POST",
                "/api/v1/documents",
                params={"provider": provider},
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            )
            st.session_state.document = response.json()
            st.rerun()
    st.divider()
    if st.button("Check backend", use_container_width=True):
        health = api_request("GET", "/health").json()
        ocr_status = "ready" if health["tesseract_available"] else "unavailable"
        st.success(f"API {health['version']} · OCR {ocr_status}")

if "document" not in st.session_state:
    st.info("Upload an invoice to begin. The rules baseline works without a paid API key.")
    st.markdown("**Review loop:** upload → inspect uncertain fields → correct → export")
    st.stop()

document = st.session_state.document
result = document["active_result"]
header_a, header_b, header_c, header_d = st.columns(4)
header_a.metric("Confidence", f"{result['overall_confidence']:.0%}")
header_b.metric("Input mode", result["extraction_method"].upper())
header_c.metric("Provider", result["provider"].title())
header_d.metric("Latency", f"{result['metrics']['latency_ms']:.0f} ms")

if result["review"]["required"]:
    st.warning("Routed to human review: " + " · ".join(result["review"]["reasons"]), icon="⚠️")
else:
    st.success("Automatic checks passed; manual review is optional.", icon="✅")

left, right = st.columns([1.1, 0.9], gap="large")
with left:
    original_tab, evidence_tab = st.tabs(["Original document", "Evidence citations"])
    with original_tab:
        render_document(document)
    with evidence_tab:
        render_evidence(document, result)

with right:
    st.subheader("Extracted fields")
    with st.form("review_form"):
        edits: dict[str, Any] = {}
        for name, label in FIELDS:
            field = result[name]
            confidence = field["confidence"]
            marker = "⚠️" if confidence < 0.75 else "✓"
            help_text = f"Confidence {confidence:.0%} · source: {field['source']}"
            if name in {"subtotal", "tax", "total"}:
                edits[name] = st.number_input(
                    f"{marker} {label}", value=float(field["value"] or 0), step=0.01, help=help_text
                )
            else:
                edits[name] = st.text_input(
                    f"{marker} {label}", value=str(field["value"] or ""), help=help_text
                )

        st.markdown("#### Line items")
        item_rows = [
            {key: item[key] for key in ("description", "quantity", "unit_price", "amount")}
            for item in result["line_items"]
        ]
        edited_items = st.data_editor(
            item_rows,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "quantity": st.column_config.NumberColumn(format="%.2f"),
                "unit_price": st.column_config.NumberColumn(format="%.2f"),
                "amount": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        saved = st.form_submit_button("Save corrections", type="primary", use_container_width=True)
        if saved:
            clean_items = []
            for item in edited_items:
                if item.get("description"):
                    clean_items.append({**item, "confidence": 1.0})
            payload = {**edits, "line_items": clean_items}
            response = api_request(
                "PUT", f"/api/v1/documents/{document['id']}/corrections", json=payload
            )
            st.session_state.document = response.json()
            st.success("Corrections saved. The original prediction remains preserved.")
            st.rerun()

    st.markdown("#### Validation")
    for validation in result["validations"]:
        if validation["passed"]:
            st.success(validation["message"], icon="✅")
        else:
            st.warning(validation["message"], icon="⚠️")

    if result["suspicion_signals"]:
        st.markdown("#### Suspicion signals")
        for signal in result["suspicion_signals"]:
            st.warning(f"{signal['severity'].upper()} · {signal['message']}")

    with st.expander("Runtime and cost"):
        metrics = result["metrics"]
        st.json(
            {
                "latency_ms": metrics["latency_ms"],
                "input_tokens": metrics["input_tokens"],
                "output_tokens": metrics["output_tokens"],
                "estimated_cost_usd": metrics["estimated_cost_usd"],
                "device": metrics["device"],
            }
        )

    st.markdown("#### Export reviewed result")
    export_json = api_request("GET", f"/api/v1/documents/{document['id']}/export?format=json")
    export_csv = api_request("GET", f"/api/v1/documents/{document['id']}/export?format=csv")
    download_a, download_b = st.columns(2)
    download_a.download_button(
        "Download JSON",
        export_json.content,
        "invoice.json",
        "application/json",
        use_container_width=True,
    )
    download_b.download_button(
        "Download CSV", export_csv.content, "invoice.csv", "text/csv", use_container_width=True
    )

with st.expander("Raw extraction payload"):
    st.json(json.loads(json.dumps(result)))
