"""
LLM pricing — the Monitor's cost numbers are ESTIMATES of paid-tier list price.

Why a table (and not two global constants): the configured model can change
(GEMINI_MODEL), and a cost computed with the wrong model's rates is silently wrong
by an order of magnitude. Every llm_calls row is stamped with the model, the
PRICING_VERSION and the estimated cost, so historical rows stay interpretable after
prices change.

Call it "estimated paid-tier cost": a free-tier project may be charged nothing,
and promotional/discounted prices end. Update PRICING (and bump PRICING_VERSION)
when Google changes list prices, or override per deployment with
LLM_INPUT_PRICE_PER_MILLION / LLM_OUTPUT_PRICE_PER_MILLION.
"""
from settings import settings

PRICING_VERSION = "2026-09-gemini-paid-tier"

# USD per 1M tokens, paid tier.
PRICING = {
    "gemini-3.6-flash": {"input_per_million": 0.75, "output_per_million": 3.75},
}


class UnknownModelPricing(KeyError):
    """No price is known for the configured model."""


def rates_for(model):
    """(input_per_million, output_per_million, pricing_version) for `model`.
    An explicit env override wins. Unknown models return (None, None, ...) so the
    cost is recorded as UNKNOWN rather than as a confident-but-wrong number."""
    if settings.llm_input_price_per_million is not None and \
            settings.llm_output_price_per_million is not None:
        return (settings.llm_input_price_per_million,
                settings.llm_output_price_per_million, "env-override")
    p = PRICING.get((model or "").strip().lower())
    if not p:
        return (None, None, PRICING_VERSION)
    return (p["input_per_million"], p["output_per_million"], PRICING_VERSION)


def estimate_cost(model, prompt_tokens, completion_tokens):
    """Estimated paid-tier USD cost of one call, and the pricing version used.
    Returns (None, version) when the model has no known price."""
    inp, out, version = rates_for(model)
    if inp is None or out is None:
        return (None, version)
    cost = ((prompt_tokens or 0) * inp + (completion_tokens or 0) * out) / 1_000_000
    return (round(cost, 8), version)
