from __future__ import annotations

from aa_agent.pii import scrub


def test_url_redacted():
    assert scrub("check http://bit.ly/abc123 for status") == "check [URL] for status"


def test_email_redacted():
    assert scrub("reach me at jane.doe@example.com please") == "reach me at [EMAIL] please"


def test_phone_redacted():
    assert "[PHONE]" in scrub("call me at 555-123-4567 asap")


def test_flight_number_survives():
    """Flight numbers are short and operationally useful -- must not be
    caught by the phone or long-digit patterns."""
    text = scrub("my flight AA1234 was delayed")
    assert "AA1234" in text
    assert "[PHONE]" not in text
    assert "[REF]" not in text


def test_confirmation_code_redacted():
    out = scrub("my confirmation number is AB12CD, please help")
    assert "AB12CD" not in out
    assert "[REF]" in out


def test_bare_long_digit_run_redacted():
    out = scrub("frequent flyer 8823910294 not showing miles")
    assert "8823910294" not in out
    assert "[REF]" in out


def test_short_digit_run_survives():
    """Gate/seat numbers etc: below the 6-digit threshold, left alone."""
    assert "12345" in scrub("gate 12345 change announced")


def test_customer_mention_redacted_brand_mention_kept():
    out = scrub("@randomuser123 told @AmericanAir about this")
    assert "@customer" in out
    assert "@AmericanAir" in out
    assert "@randomuser123" not in out


def test_brand_match_is_case_insensitive():
    out = scrub("hey @americanair any update?", brand_handles=frozenset({"americanair"}))
    assert "@americanair" in out


def test_empty_text_is_noop():
    assert scrub("") == ""


def test_combination_of_signals_in_one_message():
    raw = (
        "email me jane@x.com or call 555-999-1234, "
        "confirmation number ZX9Q2K, cc @randomguy and @AmericanAir"
    )
    out = scrub(raw)
    assert "[EMAIL]" in out
    assert "[PHONE]" in out
    assert "ZX9Q2K" not in out
    assert "@randomguy" not in out
    assert "@AmericanAir" in out
