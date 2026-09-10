"""PII scrubbing for tweet text before anything reaches an LLM API.

Heuristic, not exhaustive -- these regexes target the clearly-identifiable
signals (emails, phone numbers, URLs, booking/confirmation references,
non-brand @mentions). This is documented as a known limitation, not silently
assumed complete: see DECISIONS.md.
"""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Phone: optional country code, then groups of digits separated by common
# punctuation. Deliberately requires 7+ total digits so it doesn't eat flight
# numbers ("AA1234") or short reference codes.
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3}[\s.-]?\d{3,4}[\s.-]?\d{0,4}\b"
)

# Booking/confirmation/record-locator references: a trigger word, optionally
# "number"/"no"/"#", optionally a linking word or punctuation, then a code.
# The code group requires at least one digit -- without that, a plain label
# word like "number" (6 letters) satisfies [A-Z0-9]{5,8} under IGNORECASE
# just as well as a real code does, and gets redacted instead of the code.
_BOOKING_RE = re.compile(
    r"(?i)\b(confirmation|booking|reservation|record locator|order)\b"
    r"(?:\s*(?:number|no\.?|#))?"
    r"(?:\s*(?:is|was|[:#=]))?"
    r"\s+((?=[A-Za-z0-9]*\d)[A-Z0-9]{5,8})\b"
)

# Bare 6+ digit runs not already caught above (loyalty/frequent-flyer numbers,
# ticket numbers). Flight numbers and gate/seat references are 1-4 digits and
# pass through untouched.
_LONG_DIGITS_RE = re.compile(r"\b\d{6,}\b")

_MENTION_RE = re.compile(r"@(\w+)")


def _redact_mentions(text: str, keep: frozenset[str]) -> str:
    def repl(m: re.Match[str]) -> str:
        handle = m.group(1)
        return m.group(0) if handle.lower() in keep else "@customer"

    return _MENTION_RE.sub(repl, text)


def scrub(text: str, brand_handles: frozenset[str] = frozenset({"americanair"})) -> str:
    """Redact PII from one message. `brand_handles` are kept un-redacted
    (lowercased, no @) so the brand's own account mentions survive."""
    if not text:
        return text
    text = _URL_RE.sub("[URL]", text)
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _BOOKING_RE.sub(lambda m: f"{m.group(1)} [REF]", text)
    text = _LONG_DIGITS_RE.sub("[REF]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _redact_mentions(text, keep=brand_handles)
    return text
