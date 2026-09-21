#!/usr/bin/env python3
import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

import fitz

VENDORS = [
    "Northstar Analytics",
    "Blue Oak Supplies",
    "Nimbus Design Co",
    "Apex Office Labs",
    "Cedar Cloud Services",
]
CUSTOMERS = [
    "Acme Retail Ltd",
    "Orbit Foods Pvt Ltd",
    "Helios Studio",
    "Maple Works Inc",
    "Vertex Education",
]
ITEMS = [
    ("Data analysis workshop", 450.0),
    ("Premium paper boxes", 32.5),
    ("UX research session", 275.0),
    ("Cloud support hours", 95.0),
    ("Document digitization", 18.75),
    ("Annual software license", 720.0),
]
CURRENCIES = [("USD", "$"), ("EUR", "€"), ("GBP", "£"), ("INR", "₹")]


def build_invoice(index: int, rng: random.Random) -> dict:
    currency, symbol = CURRENCIES[index % len(CURRENCIES)]
    line_items = []
    for description, price in rng.sample(ITEMS, rng.randint(2, 4)):
        quantity = rng.randint(1, 5)
        line_items.append(
            {
                "description": description,
                "quantity": quantity,
                "unit_price": price,
                "amount": round(quantity * price, 2),
            }
        )
    subtotal = round(sum(item["amount"] for item in line_items), 2)
    tax_rate = [0.0, 0.05, 0.1, 0.18][index % 4]
    tax = round(subtotal * tax_rate, 2)
    invoice_date = date(2025, 1, 1) + timedelta(days=index * 7)
    return {
        "invoice_number": f"DM-{2025 + index // 24}-{1001 + index}",
        "invoice_date": invoice_date.isoformat(),
        "vendor_name": VENDORS[index % len(VENDORS)],
        "customer_name": CUSTOMERS[(index * 3) % len(CUSTOMERS)],
        "currency": currency,
        "currency_symbol": symbol,
        "subtotal": subtotal,
        "tax": tax,
        "total": round(subtotal + tax, 2),
        "line_items": line_items,
    }


def draw_invoice(invoice: dict, path: Path, variant: int, scanned: bool) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    color = (0.08, 0.18, 0.32)
    page.insert_text((50, 62), invoice["vendor_name"], fontsize=19, color=color)
    page.insert_text((50, 84), "INVOICE", fontsize=10, color=(0.4, 0.4, 0.4))
    labels = [
        ("Invoice No:", invoice["invoice_number"]),
        ("Invoice Date:", invoice["invoice_date"]),
        ("Bill To:", invoice["customer_name"]),
    ]
    y = 125
    for label, value in labels:
        page.insert_text((50, y), f"{label} {value}", fontsize=11)
        y += 20
    page.draw_line((50, 205), (545, 205), color=color, width=1)
    if variant % 2:
        page.insert_text((50, 225), "Description | Qty | Unit Price | Amount", fontsize=10)
    else:
        page.insert_text(
            (50, 225), "Description                         Qty   Unit Price   Amount", fontsize=10
        )
    y = 252
    symbol = invoice["currency_symbol"]
    # The built-in PDF fonts do not contain every currency glyph. ISO codes
    # keep generated documents portable when a glyph is unavailable.
    money_marker = symbol if symbol in {"$", "£"} else f"{invoice['currency']} "
    for item in invoice["line_items"]:
        if variant % 2:
            row = (
                f"{item['description']} | {item['quantity']} | "
                f"{money_marker}{item['unit_price']:.2f} | "
                f"{money_marker}{item['amount']:.2f}"
            )
        else:
            row = (
                f"{item['description']:<34}  {item['quantity']:<4}  "
                f"{money_marker}{item['unit_price']:<10.2f}  "
                f"{money_marker}{item['amount']:.2f}"
            )
        page.insert_text((50, y), row, fontsize=9, fontname="cour")
        y += 24
    y += 20
    page.insert_text((330, y), f"Subtotal: {money_marker}{invoice['subtotal']:.2f}", fontsize=11)
    page.insert_text((330, y + 22), f"Tax: {money_marker}{invoice['tax']:.2f}", fontsize=11)
    page.insert_text(
        (330, y + 50),
        f"Total: {money_marker}{invoice['total']:.2f}",
        fontsize=13,
        color=color,
    )
    page.insert_text(
        (50, 780), "Thank you for your business.", fontsize=9, color=(0.45, 0.45, 0.45)
    )

    if scanned:
        pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), colorspace=fitz.csGRAY, alpha=False)
        image_bytes = pix.tobytes("jpeg", jpg_quality=72)
        scanned_doc = fitz.open()
        scanned_page = scanned_doc.new_page(width=595, height=842)
        scanned_page.insert_image(scanned_page.rect, stream=image_bytes)
        scanned_doc.save(path, deflate=True)
        scanned_doc.close()
    else:
        doc.save(path, deflate=True)
    doc.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic invoice PDFs")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("evaluation/generated"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scan-every",
        type=int,
        default=3,
        help="Render every Nth PDF as image-only; use 0 for a text-only corpus.",
    )
    parser.add_argument(
        "--anomaly-every",
        type=int,
        default=0,
        help="Make every Nth invoice arithmetically inconsistent; use 0 for none.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    records = []
    for index in range(args.count):
        invoice = build_invoice(index, rng)
        scanned = args.scan_every > 0 and (index + 1) % args.scan_every == 0
        suspicious = args.anomaly_every > 0 and (index + 1) % args.anomaly_every == 0
        if suspicious:
            invoice["total"] = round(invoice["total"] + max(17.35, invoice["total"] * 0.08), 2)
        filename = f"invoice_{index + 1:03d}_{'scan' if scanned else 'text'}.pdf"
        draw_invoice(invoice, args.output / filename, index, scanned)
        invoice.pop("currency_symbol")
        records.append(
            {
                "filename": filename,
                "scanned": scanned,
                "suspicious": suspicious,
                "ground_truth": invoice,
            }
        )
    (args.output / "ground_truth.json").write_text(json.dumps(records, indent=2))
    print(f"Generated {len(records)} invoices in {args.output}")


if __name__ == "__main__":
    main()
