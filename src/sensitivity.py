"""
One-at-a-time sensitivity analysis around a base scenario.

For each selected input parameter, the value is varied by +/- a given
percentage while holding all other inputs fixed, and the resulting NPV
is recorded. This produces the data behind a standard "tornado chart":
which assumptions the ROI conclusion is most sensitive to.
"""

from dataclasses import replace
from roi_model import ScenarioInputs, build_cash_flows, npv


# Parameters worth stress-testing, and the +/- swing to apply (as a fraction)
DEFAULT_SWING = 0.25

PARAMETERS = [
    "undetected_loss_pct",
    "recovery_effectiveness_pct",
    "electricity_price_eur_per_kwh",
    "inspection_cost_eur",
    "discount_rate_pct",
]


def _vary(inputs: ScenarioInputs, field: str, factor: float) -> ScenarioInputs:
    current = getattr(inputs, field)
    return replace(inputs, **{field: current * factor})


def run_sensitivity(inputs: ScenarioInputs, swing: float = DEFAULT_SWING) -> list:
    """
    Returns a list of dicts: [{parameter, low_npv, base_npv, high_npv, range}, ...]
    sorted by impact range (largest first) — ready for a tornado chart.
    """
    base_flows = build_cash_flows(inputs)
    base_npv = npv(base_flows, inputs.discount_rate_pct)

    results = []
    for field in PARAMETERS:
        low_inputs = _vary(inputs, field, 1 - swing)
        high_inputs = _vary(inputs, field, 1 + swing)

        low_npv = npv(build_cash_flows(low_inputs), low_inputs.discount_rate_pct)
        high_npv = npv(build_cash_flows(high_inputs), high_inputs.discount_rate_pct)

        results.append({
            "parameter": field,
            "low_npv": low_npv,
            "base_npv": base_npv,
            "high_npv": high_npv,
            "range": abs(high_npv - low_npv),
        })

    results.sort(key=lambda r: r["range"], reverse=True)
    return results
