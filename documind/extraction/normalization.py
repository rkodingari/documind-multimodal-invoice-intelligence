import re
from datetime import date

from dateutil import parser as date_parser

CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "¥": "JPY"}
CURRENCY_ALIASES = {
    "US$": "USD",
    "USD": "USD",
    "EUR": "EUR",
    "GBP": "GBP",
    "INR": "INR",
    "JPY": "JPY",
    "CAD": "CAD",
    "AUD": "AUD",
}


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t:|")


def normalize_date(value: str) -> str | None:
    cleaned = normalize_whitespace(value)
    try:
        parsed: date = date_parser.parse(cleaned, dayfirst=False, fuzzy=False).date()
        return parsed.isoformat()
    except (ValueError, OverflowError):
        try:
            parsed = date_parser.parse(cleaned, dayfirst=True, fuzzy=False).date()
            return parsed.isoformat()
        except (ValueError, OverflowError):
            return None


def normalize_number(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return round(float(value), 2)
    cleaned = value.strip().replace(" ", "")
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = re.sub(r"[^0-9,.-]", "", cleaned)
    if not cleaned:
        return None
    if "," in cleaned and "." not in cleaned:
        parts = cleaned.split(",")
        if len(parts[-1]) == 2:
            cleaned = ".".join(parts)
        else:
            cleaned = "".join(parts)
    else:
        cleaned = cleaned.replace(",", "")
    try:
        number = float(cleaned)
        return round(-number if negative else number, 2)
    except ValueError:
        return None


def detect_currency(text: str) -> tuple[str | None, str | None]:
    for code in CURRENCY_ALIASES:
        if re.search(rf"(?<![A-Z]){re.escape(code)}(?![A-Z])", text, re.IGNORECASE):
            return CURRENCY_ALIASES[code], code
    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code, symbol
    return None, None
