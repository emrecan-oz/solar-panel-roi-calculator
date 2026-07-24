"""
Core financial model for evaluating the ROI of periodic drone-based
thermal inspection of a PV system.

IMPORTANT: All numbers used with this model in this repository are
illustrative / assumption-driven examples. This is a generic, openly
published version of a financial modeling methodology, not a specific
client's or company's proprietary model or data.

Modeling idea
--------------
A PV system produces an expected amount of energy each year. Without
inspection, undetected faults (hotspots, bypass diode failures, string
outages, soiling) cause a portion of that production to be silently
lost. Periodic drone thermal inspection finds these faults early so
they can be repaired, recovering most of that lost production.

The inspection is treated as a recurring investment:
  - Cost: the price of a drone inspection, paid every `interval_years`.
  - Benefit: the value of the production loss avoided each year because
    faults were caught and fixed instead of running undetected.

The resulting cash flow series is used to compute NPV, IRR and a
simple payback period, exactly like any other capital investment
decision.
"""

from dataclasses import dataclass


@dataclass
class ScenarioInputs:
    system_kwp: float                  # installed system size
    specific_yield_kwh_per_kwp: float   # expected annual yield per kWp (location-dependent)
    electricity_price_eur_per_kwh: float  # feed-in tariff or market price
    undetected_loss_pct: float         # annual production lost to undetected faults (0-1)
    recovery_effectiveness_pct: float  # share of that loss avoided thanks to inspection+repair (0-1)
    inspection_cost_eur: float         # cost per inspection event
    interval_years: int                # years between inspections
    discount_rate_pct: float           # annual discount rate for NPV/IRR (0-1)
    horizon_years: int                 # analysis horizon


def annual_expected_production_kwh(inputs: ScenarioInputs) -> float:
    return inputs.system_kwp * inputs.specific_yield_kwh_per_kwp


def annual_benefit_eur(inputs: ScenarioInputs) -> float:
    """Value of production loss avoided each year thanks to inspection."""
    production = annual_expected_production_kwh(inputs)
    avoided_loss_kwh = production * inputs.undetected_loss_pct * inputs.recovery_effectiveness_pct
    return avoided_loss_kwh * inputs.electricity_price_eur_per_kwh


def build_cash_flows(inputs: ScenarioInputs) -> list:
    """
    Returns a list of length horizon_years + 1 (year 0 .. horizon_years).
    Year 0: initial inspection cost only (no benefit realized yet).
    Later years: annual benefit, minus a re-inspection cost on interval years.
    """
    benefit = annual_benefit_eur(inputs)
    flows = [-inputs.inspection_cost_eur]  # year 0

    for year in range(1, inputs.horizon_years + 1):
        flow = benefit
        if year % inputs.interval_years == 0:
            flow -= inputs.inspection_cost_eur
        flows.append(flow)

    return flows


def npv(cash_flows: list, discount_rate_pct: float) -> float:
    return sum(cf / (1 + discount_rate_pct) ** t for t, cf in enumerate(cash_flows))


def irr(cash_flows: list, scan_low: float = 0.0, scan_high: float = 3.0, scan_step: float = 0.02,
        tol: float = 1e-6, max_iter: int = 200):
    """
    Internal rate of return. Scans discount rates from scan_low to
    scan_high for a sign change in NPV, then refines the root via
    bisection. The scan is deliberately restricted to economically
    plausible discount rates (default 0%-300%): for long horizons,
    NPV(rate) blows up numerically as rate approaches -100% (the
    discount factor (1+rate)^-t explodes), which can produce spurious
    sign changes at deeply negative, meaningless rates. Returns None if
    no sign change is found in the scanned range.
    """
    def f(rate):
        return npv(cash_flows, rate)

    low = scan_low
    f_low = f(low)
    bracket = None
    rate = scan_low
    while rate < scan_high:
        next_rate = rate + scan_step
        f_next = f(next_rate)
        if f_low * f_next < 0:
            bracket = (rate, next_rate, f_low, f_next)
            break
        rate, f_low = next_rate, f_next

    if bracket is None:
        return None

    low, high, f_low, f_high = bracket
    for _ in range(max_iter):
        mid = (low + high) / 2
        f_mid = f(mid)
        if abs(f_mid) < tol:
            return mid
        if f_low * f_mid < 0:
            high = mid
        else:
            low, f_low = mid, f_mid

    return (low + high) / 2


def payback_period_years(cash_flows: list):
    """First year at which cumulative cash flow turns non-negative. None if never."""
    cumulative = 0.0
    for year, cf in enumerate(cash_flows):
        cumulative += cf
        if cumulative >= 0:
            return year
    return None


def summarize(inputs: ScenarioInputs) -> dict:
    flows = build_cash_flows(inputs)
    return {
        "annual_expected_production_kwh": annual_expected_production_kwh(inputs),
        "annual_benefit_eur": annual_benefit_eur(inputs),
        "cash_flows": flows,
        "npv_eur": npv(flows, inputs.discount_rate_pct),
        "irr_pct": irr(flows),
        "payback_years": payback_period_years(flows),
    }
