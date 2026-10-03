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
import math
from datetime import datetime, timedelta, timezone

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
    inp_o = settings.llm_input_price_per_million
    out_o = settings.llm_output_price_per_million
    if inp_o is not None and out_o is not None:
        return (float(inp_o), float(out_o), "env-override")
    if (inp_o is None) != (out_o is None):
        # Half an override (settings.py rejects this at startup; this is the
        # backstop for code that changes settings at runtime): the operator's
        # intent is unknowable, so the price is UNKNOWN — fail closed.
        return (None, None, None)
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


# A request priced at dispatch can complete after a price change. The reservation
# uses the HIGHEST rate in effect at any moment of this window after dispatch.
PRICE_CHANGE_HORIZON = timedelta(minutes=15)


def byte_token_bound(prompt_bytes):
    """Proven upper bound on input tokens: one token never covers less than one
    UTF-8 byte, so bytes / bytes_per_token (<= 1.0, enforced by settings) + 1."""
    bpt = float(getattr(settings, "llm_reserve_bytes_per_token", 1.0) or 1.0)
    bpt = min(bpt, 1.0)            # backstop: never trust a value that under-reserves
    return int(prompt_bytes / bpt) + 1


def max_request_cost(model, prompt_bytes, max_output_tokens, at=None, input_tokens=None):
    """Conservative UPPER BOUND on one request's cost, used to reserve dollars
    before dispatch. Input tokens: `input_tokens` when the caller has a tighter
    bound (llm.input_token_bound — a provider count with a margin, itself capped
    by the byte bound), else the byte bound. Output: the max_output_tokens the
    request itself enforces (thinking tokens count against it). Priced at the
    highest rate in [at, at + PRICE_CHANGE_HORIZON]. None if the price is unknown."""
    at = at or _now()
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    rates = [rates_for(model, at), rates_for(model, at + PRICE_CHANGE_HORIZON)]
    if any(r[0] is None or r[1] is None for r in rates):
        return None
    inp = max(r[0] for r in rates)
    out = max(r[1] for r in rates)
    bound = byte_token_bound(prompt_bytes)
    in_tokens = bound if input_tokens is None else min(int(input_tokens), bound)
    raw = (in_tokens * inp + int(max_output_tokens) * out) / 1_000_000
    # Round UP so rounding can never shave the bound below the true maximum.
    return math.ceil(raw * 1e8) / 1e8


def price_known(model=None, at=None):
    """True if a cost can be computed for `model` (default: the configured model)
    now. A cost-bounded run must not make model calls whose cost it cannot bound."""
    inp, out, _version = rates_for(model if model is not None else settings.gemini_model, at)
    return inp is not None and out is not None