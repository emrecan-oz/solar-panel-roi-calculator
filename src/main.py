import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from roi_model import ScenarioInputs, summarize
from sensitivity import run_sensitivity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "example_scenario.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")


def load_scenario(path: str) -> ScenarioInputs:
    with open(path) as f:
        raw = json.load(f)
    raw.pop("_comment", None)
    return ScenarioInputs(**raw)


def plot_cash_flows(cash_flows, path):
    years = list(range(len(cash_flows)))
    cumulative = []
    running = 0
    for cf in cash_flows:
        running += cf
        cumulative.append(running)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    colors = ["#c0392b" if cf < 0 else "#2e7d32" for cf in cash_flows]
    ax1.bar(years, cash_flows, color=colors, label="Annual net cash flow")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Annual net cash flow (EUR)")
    ax1.axhline(0, color="black", linewidth=0.8)

    ax2 = ax1.twinx()
    ax2.plot(years, cumulative, color="#1f4e79", marker="o", markersize=3, label="Cumulative cash flow")
    ax2.set_ylabel("Cumulative cash flow (EUR)")

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.suptitle("Drone Inspection: Annual and Cumulative Cash Flow (illustrative example)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tornado(sensitivity_results, path):
    labels = [r["parameter"].replace("_", " ") for r in sensitivity_results]
    base = sensitivity_results[0]["base_npv"]
    lows = [r["low_npv"] - base for r in sensitivity_results]
    highs = [r["high_npv"] - base for r in sensitivity_results]

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = range(len(labels))

    for i, (lo, hi) in enumerate(zip(lows, highs)):
        left = min(lo, hi)
        width = abs(hi - lo)
        color = "#2e7d32" if hi >= lo else "#c0392b"
        ax.barh(i, width, left=left, color="#5b8db8", edgecolor="black", linewidth=0.5)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Change in NPV vs. base case (EUR)")
    ax.set_title("Sensitivity of NPV to Key Assumptions (\u00b125% swing, illustrative)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = load_scenario(DATA_PATH)
    result = summarize(inputs)
    sens = run_sensitivity(inputs)

    print("=== Drone Inspection ROI Model (illustrative example) ===")
    print(f"System size:              {inputs.system_kwp:,.0f} kWp")
    print(f"Expected annual production: {result['annual_expected_production_kwh']:,.0f} kWh")
    print(f"Annual benefit (avoided loss): EUR {result['annual_benefit_eur']:,.0f}")
    print(f"Inspection cost per event: EUR {inputs.inspection_cost_eur:,.0f} every {inputs.interval_years} years")
    print(f"Discount rate:            {inputs.discount_rate_pct * 100:.1f}%")
    print(f"Horizon:                  {inputs.horizon_years} years")
    print("---")
    print(f"NPV:                      EUR {result['npv_eur']:,.0f}")
    irr_txt = f"{result['irr_pct'] * 100:.1f}%" if result["irr_pct"] is not None else "n/a (no sign change)"
    print(f"IRR:                      {irr_txt}")
    payback_txt = f"{result['payback_years']} years" if result["payback_years"] is not None else "not within horizon"
    print(f"Payback period:           {payback_txt}")

    cash_flow_path = os.path.join(OUTPUT_DIR, "cash_flow_chart.png")
    tornado_path = os.path.join(OUTPUT_DIR, "sensitivity_tornado.png")
    plot_cash_flows(result["cash_flows"], cash_flow_path)
    plot_tornado(sens, tornado_path)

    print("---")
    print(f"Charts written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
