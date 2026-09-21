import io
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

from documind.schemas import InvoicePrediction

FIELD_COLORS = {
    "invoice_number": "#2563EB",
    "invoice_date": "#7C3AED",
    "vendor_name": "#059669",
    "customer_name": "#0D9488",
    "currency": "#D97706",
    "subtotal": "#EA580C",
    "tax": "#DC2626",
    "total": "#BE123C",
    "line_item": "#4F46E5",
}


def render_page(
    path: Path, content_type: str, page_number: int
) -> tuple[Image.Image, float, float]:
    if content_type == "application/pdf" or path.suffix.lower() == ".pdf":
        document = fitz.open(path)
        try:
            if not 1 <= page_number <= document.page_count:
                raise ValueError("Page number is out of range")
            page = document[page_number - 1]
            scale = 1.7
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            return image, scale, scale
        finally:
            document.close()
    if page_number != 1:
        raise ValueError("Page number is out of range")
    return Image.open(path).convert("RGB"), 1.0, 1.0


def render_evidence(
    path: Path, content_type: str, prediction: InvoicePrediction, page_number: int
) -> bytes:
    image, x_scale, y_scale = render_page(path, content_type, page_number)
    draw = ImageDraw.Draw(image, "RGBA")
    entries = []
    for name in (
        "invoice_number",
        "invoice_date",
        "vendor_name",
        "customer_name",
        "currency",
        "subtotal",
        "tax",
        "total",
    ):
        for evidence in getattr(prediction, name).evidence:
            entries.append((name, evidence))
    for item in prediction.line_items:
        for evidence in item.evidence:
            entries.append(("line_item", evidence))
    for name, evidence in entries:
        if evidence.page != page_number:
            continue
        x0, y0, x1, y1 = evidence.bbox
        box = (x0 * x_scale, y0 * y_scale, x1 * x_scale, y1 * y_scale)
        color = FIELD_COLORS[name]
        draw.rectangle(box, outline=color, width=4)
        label_y = max(0, box[1] - 18)
        draw.rectangle((box[0], label_y, box[0] + max(70, len(name) * 7), box[1]), fill=color)
        draw.text((box[0] + 3, label_y + 2), name, fill="white")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
