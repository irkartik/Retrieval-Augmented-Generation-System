"""Approximate per-query API cost, for the operational trade-off metric (master plan §5).

Prices are USD per 1,000 tokens (input, output), current as of Sep 2026. These are
list prices for rough comparison across strategies, not billing — update the table if
your rate card differs. estimate_cost_usd returns None for unknown models so the harness
records a null cost rather than a wrong one.
"""

# model name (or prefix) -> (usd_per_1k_input, usd_per_1k_output)
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o": (0.00250, 0.01000),
    "openai.gpt-oss-120": (0.00015, 0.00060),
    "claude-haiku-4-5": (0.00080, 0.00400),
    "claude-3-5-haiku": (0.00080, 0.00400),
    "claude-3-haiku": (0.00025, 0.00125),
}


def _match_model(model: str) -> str | None:
    if model in PRICING:
        return model
    # fall back to the longest known prefix (handles dated suffixes like -20251001)
    candidates = [k for k in PRICING if model.startswith(k)]
    return max(candidates, key=len) if candidates else None


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    key = _match_model(model)
    if key is None:
        return None
    per_in, per_out = PRICING[key]
    return (prompt_tokens / 1000.0) * per_in + (completion_tokens / 1000.0) * per_out
