import pytest

from documind.extraction.normalization import detect_currency, normalize_date, normalize_number


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("$1,234.50", 1234.5), ("1 234,50", 1234.5), ("(25.00)", -25.0), (None, None)],
)
def test_normalize_number(raw, expected):
    assert normalize_number(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("2025-04-30", "2025-04-30"), ("April 30, 2025", "2025-04-30"), ("not-a-date", None)],
)
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected


def test_detect_currency_prefers_iso_code():
    assert detect_currency("Currency: EUR Total €45.00") == ("EUR", "EUR")
