"""
LLM pricing — the Monitor's cost numbers are ESTIMATES of paid-tier list price.

Prices change on known dates, so the table is EFFECTIVE-DATED: each model has a
list of price windows [valid_from, valid_until), and a call is priced with the
window that contains the moment it was made. A timeless dictionary silently
becomes wrong the day a price changes; this table does not.

Every llm_calls row is stamped with the model, the pricing_version of the window
used and the estimated cost, so historical rows stay interpretable.

"Output" is billed INCLUDING thinking tokens (Gemini bills thoughts at the output
rate), so callers must pass candidates + thoughts as completion tokens.

Call it "estimated paid-tier cost": a free-tier project may be charged nothing.
Add a new window (never edit an old one) when Google publishes a price change, or
override per deployment with LLM_INPUT_PRICE_PER_MILLION / LLM_OUTPUT_PRICE_PER_MILLION.

Source for gemini-3.6-flash (checked 2026-09-29, ai.google.dev/gemini-api/docs/pricing):
  launched 2026-07-21 at $1.50 / $7.50; introductory $0.75 / $3.75 from 2026-08-13
  through 2026-12-31; $1.50 / $7.50 from 2027-01-01.
"""
from dataclasses import dataclass
from datetime import datetime, timezone

from settings import settings


def _d(y, m, d):
    return datetime(y, m, d, tzinfo=timezone.utc)


@dataclass(frozen=True)
class PriceWindow:
    valid_from: datetime                 # inclusive
    valid_until: datetime | None         # exclusive; None = open-ended
    input_per_million: float
    output_per_million: float            # includes thinking tokens
    version: str


# USD per 1M tokens, paid tier, standard processing. Windows per model must not
# overlap (enforced at import by _validate()).
PRICING = {
    "gemini-3.6-flash": (
        PriceWindow(_d(2026, 7, 21), _d(2026, 8, 13), 1.50, 7.50, "gemini-3.6-flash@2026-07-21"),
        PriceWindow(_d(2026, 8, 13), _d(2027, 1, 1), 0.75, 3.75, "gemini-3.6-flash@2026-08-13"),
        PriceWindow(_d(2027, 1, 1), None, 1.50, 7.50, "gemini-3.6-flash@2027-01-01"),
    ),
}


def _validate():
    for model, windows in PRICING.items():
        ordered = sorted(windows, key=lambda w: w.valid_from)
        for a, b in zip(ordered, ordered[1:]):
            if a.valid_until is None or a.valid_until > b.valid_from:
                raise ValueError(f"pricing windows overlap for {model}: {a.version} / {b.version}")


_validate()


class UnknownModelPricing(KeyError):
    """No price is known for the configured model."""


def _now():
    return datetime.now(timezone.utc)


def rates_for(model, at=None):
    """(input_per_million, output_per_million, pricing_version) for `model` at the
    instant `at` (default: now). An explicit env override wins. A model with no
    window covering `at` returns (None, None, None) so the cost is recorded as
    UNKNOWN rather than as a confident-but-wrong number."""
    if settings.llm_input_price_per_million is not None and \
            settings.llm_output_price_per_million is not None:
        return (settings.llm_input_price_per_million,
                settings.llm_output_price_per_million, "env-override")
    at = at or _now()
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    for w in PRICING.get((model or "").strip().lower(), ()):
        if w.valid_from <= at and (w.valid_until is None or at < w.valid_until):
            return (w.input_per_million, w.output_per_million, w.version)
    return (None, None, None)


def estimate_cost(model, prompt_tokens, completion_tokens, at=None):
    """Estimated paid-tier USD cost of one call, and the pricing version used.

    Returns (None, version) when the model has no known price at `at`, AND when a
    token count is missing (None): a call whose usage is unknown has an unknown
    cost — it is never silently priced as zero tokens."""
    inp, out, version = rates_for(model, at)
    if inp is None or out is None or prompt_tokens is None or completion_tokens is None:
        return (None, version)
    cost = (int(prompt_tokens) * inp + int(completion_tokens) * out) / 1_000_000
    return (round(cost, 8), version)


def max_request_cost(model, prompt_bytes, max_output_tokens, at=None):
    """Conservative UPPER BOUND on one request's cost, used to reserve dollars
    before dispatch. Input tokens are bounded by the prompt's UTF-8 byte length
    divided by settings.llm_reserve_bytes_per_token (default 1.0: a tokenizer never
    produces more tokens than bytes); output is bounded by the max_output_tokens the
    request itself enforces (thinking tokens count against it). None if the price
    is unknown."""
    inp, out, _v = rates_for(model, at)
    if inp is None or out is None:
        return None
    bpt = float(getattr(settings, "llm_reserve_bytes_per_token", 1.0) or 1.0)
    in_tokens = int(prompt_bytes / bpt) + 1
    return round((in_tokens * inp + int(max_output_tokens) * out) / 1_000_000, 8)


def price_known(model=None, at=None):
    """True if a cost can be computed for `model` (default: the configured model)
    now. A cost-bounded run must not make model calls whose cost it cannot bound."""
    inp, out, _version = rates_for(model if model is not None else settings.gemini_model, at)
    return inp is not None and out is not None