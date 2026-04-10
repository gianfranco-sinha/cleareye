"""Drift detection experiment on cleaned Session 4 data.

Analyses the 3-day continuous recording (Session 4, Oct 25–28) to detect
sensor fouling.  Strategy:

1. Identify "stable" windows where temperature and TDS variance is low
   (environmental conditions unchanged).
2. Track the rolling median of turbidity_adc within those windows.
3. Fit a linear trend to the stable-window medians — a statistically
   significant positive slope (ADC rising = signal attenuating) indicates
   biofouling on the SEN0189 optical window.
4. Output a multi-panel graph showing raw signal, rolling baseline,
   detected drift, and a cleaning-alert threshold.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).resolve().parent / "data01_clean.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "drift_detection.png"

WINDOW_MINUTES = 30          # rolling window size
TEMP_STABILITY_STD = 0.3     # max temp stdev within window to count as "stable"
TDS_STABILITY_STD = 15.0     # max TDS stdev within window
TURB_STABILITY_STD = 60.0    # max turbidity stdev within window (reject transients)
DRIFT_ALERT_SLOPE = 0.5      # ADC units / hour — threshold for cleaning alert
SESSION = "4"

# Focus on the high-turbidity plateau for drift analysis
# (the regime transition on Oct 26 ~12:00 swamps any real fouling signal)
PLATEAU_START = datetime.fromisoformat("2024-10-26T14:00:00")
PLATEAU_END = datetime.fromisoformat("2024-10-28T10:00:00")


def load_session(path: Path, session: str) -> list[dict]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return [
            {
                "ts": datetime.fromisoformat(r["timestamp"]),
                "turb": int(r["turbidity_adc"]),
                "temp": float(r["water_temperature"]),
                "tds": int(r["tds"]),
            }
            for r in reader
            if r["session"] == session
        ]


def rolling_windows(
    rows: list[dict], window_min: int
) -> list[dict]:
    """Compute rolling statistics over fixed-duration windows."""
    if not rows:
        return []

    results = []
    start = rows[0]["ts"]
    end = rows[-1]["ts"]
    delta = timedelta(minutes=window_min)
    cursor = start

    while cursor + delta <= end:
        win_end = cursor + delta
        subset = [r for r in rows if cursor <= r["ts"] < win_end]
        if len(subset) >= 10:
            turbs = np.array([r["turb"] for r in subset])
            temps = np.array([r["temp"] for r in subset])
            tdss = np.array([r["tds"] for r in subset])
            results.append({
                "ts_center": cursor + delta / 2,
                "turb_median": float(np.median(turbs)),
                "turb_mean": float(np.mean(turbs)),
                "turb_std": float(np.std(turbs)),
                "temp_std": float(np.std(temps)),
                "tds_std": float(np.std(tdss)),
                "temp_mean": float(np.mean(temps)),
                "tds_mean": float(np.mean(tdss)),
                "n": len(subset),
            })
        cursor += timedelta(minutes=window_min // 2)  # 50% overlap

    return results


def detect_drift(windows: list[dict]) -> dict:
    """Fit linear trend to stable-window turbidity medians.

    Returns dict with slope (ADC/hour), intercept, r², and per-window
    arrays for plotting.
    """
    stable = [
        w for w in windows
        if w["temp_std"] < TEMP_STABILITY_STD
        and w["tds_std"] < TDS_STABILITY_STD
        and w["turb_std"] < TURB_STABILITY_STD
    ]
    if len(stable) < 3:
        return {"slope": 0.0, "r2": 0.0, "stable_windows": [], "trend_ts": [], "trend_vals": []}

    t0 = stable[0]["ts_center"]
    hours = np.array([(w["ts_center"] - t0).total_seconds() / 3600 for w in stable])
    medians = np.array([w["turb_median"] for w in stable])

    # Linear regression
    coeffs = np.polyfit(hours, medians, 1)
    slope, intercept = coeffs
    trend_vals = np.polyval(coeffs, hours)
    ss_res = np.sum((medians - trend_vals) ** 2)
    ss_tot = np.sum((medians - np.mean(medians)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "stable_windows": stable,
        "trend_hours": hours,
        "trend_vals": trend_vals,
    }


def plot(
    rows: list[dict],
    all_windows: list[dict],
    plateau_windows: list[dict],
    drift: dict,
    output: Path,
) -> None:
    """Generate a 3-panel drift detection figure."""
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    fig.suptitle("Sensor Drift Detection — Session 4 (Oct 25–28 2024)", fontsize=14, fontweight="bold")

    ts_all = [r["ts"] for r in rows]
    turb_all = [r["turb"] for r in rows]

    # ── Panel 1: Full session overview with plateau region highlighted ─
    ax1 = axes[0]
    ax1.scatter(ts_all, turb_all, s=0.3, alpha=0.15, color="steelblue", label="Raw ADC")
    win_ts = [w["ts_center"] for w in all_windows]
    win_med = [w["turb_median"] for w in all_windows]
    ax1.plot(win_ts, win_med, color="darkorange", linewidth=1.5, label=f"Rolling median ({WINDOW_MINUTES}min)")
    ax1.axvspan(PLATEAU_START, PLATEAU_END, alpha=0.08, color="red", label="Drift analysis window")
    ax1.set_ylabel("Turbidity ADC")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Full Session — Raw Signal & Rolling Median")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Plateau zoom — stable windows + drift trend ─────────
    ax2 = axes[1]
    # Plot all plateau windows as context
    plat_ts = [w["ts_center"] for w in plateau_windows]
    plat_med = [w["turb_median"] for w in plateau_windows]
    ax2.scatter(plat_ts, plat_med, s=8, color="lightgrey", alpha=0.6, label="All plateau windows")

    stable = drift["stable_windows"]
    if stable:
        stable_ts = [w["ts_center"] for w in stable]
        stable_med = [w["turb_median"] for w in stable]
        ax2.scatter(stable_ts, stable_med, s=14, color="seagreen", zorder=3, label="Stable windows")

        # Trend line
        ax2.plot(stable_ts, drift["trend_vals"], color="red", linewidth=2.5,
                 label=f"Drift: {drift['slope']:+.2f} ADC/hr (R²={drift['r2']:.3f})")

        alert = abs(drift["slope"]) > DRIFT_ALERT_SLOPE
        alert_text = "CLEANING ALERT" if alert else "Within tolerance"
        alert_color = "red" if alert else "green"
        ax2.text(0.98, 0.05, alert_text, transform=ax2.transAxes,
                 fontsize=12, fontweight="bold", color=alert_color,
                 ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=alert_color, alpha=0.9))

        # Annotate total drift
        total_hours = drift["trend_hours"][-1] - drift["trend_hours"][0]
        total_drift = drift["slope"] * total_hours
        ax2.text(0.98, 0.18, f"Total drift: {total_drift:+.0f} ADC over {total_hours:.0f}h",
                 transform=ax2.transAxes, fontsize=10, ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.9))

    ax2.set_ylabel("Turbidity ADC")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_title(f"Drift Analysis — High-Turbidity Plateau ({PLATEAU_START.strftime('%b %d %H:%M')} → {PLATEAU_END.strftime('%b %d %H:%M')})")
    ax2.set_xlim(PLATEAU_START, PLATEAU_END)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Environmental context for plateau ───────────────────
    ax3 = axes[2]
    plat_rows = [r for r in rows if PLATEAU_START <= r["ts"] <= PLATEAU_END]
    plat_ts_raw = [r["ts"] for r in plat_rows]
    plat_temp = [r["temp"] for r in plat_rows]
    plat_tds = [r["tds"] for r in plat_rows]

    ax3.scatter(plat_ts_raw, plat_temp, s=0.3, alpha=0.2, color="coral", label="Temperature (°C)")
    ax3_twin = ax3.twinx()
    ax3_twin.scatter(plat_ts_raw, plat_tds, s=0.3, alpha=0.2, color="mediumpurple", label="TDS (ppm)")
    ax3.set_ylabel("Temperature (°C)", color="coral")
    ax3_twin.set_ylabel("TDS (ppm)", color="mediumpurple")
    ax3.set_xlabel("Time")

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax3.set_title("Environmental Context — Plateau Period")
    ax3.set_xlim(PLATEAU_START, PLATEAU_END)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax3.grid(True, alpha=0.3)

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


def main() -> None:
    print("Loading Session 4 data...")
    rows = load_session(DATA_PATH, SESSION)
    print(f"  {len(rows):,} rows loaded")

    print("Computing rolling windows (full session)...")
    all_windows = rolling_windows(rows, WINDOW_MINUTES)
    print(f"  {len(all_windows)} windows total")

    # Focus on the high-turbidity plateau for drift analysis
    plateau_rows = [r for r in rows if PLATEAU_START <= r["ts"] <= PLATEAU_END]
    print(f"  Plateau subset: {len(plateau_rows):,} rows ({PLATEAU_START} → {PLATEAU_END})")

    print("Computing rolling windows (plateau)...")
    plateau_windows = rolling_windows(plateau_rows, WINDOW_MINUTES)
    print(f"  {len(plateau_windows)} plateau windows")

    print("Detecting drift in stable plateau windows...")
    drift = detect_drift(plateau_windows)
    stable_count = len(drift["stable_windows"])
    total_count = len(plateau_windows)
    print(f"  {stable_count}/{total_count} plateau windows classified as stable")
    print(f"  Drift slope: {drift['slope']:+.3f} ADC/hour")
    print(f"  R²: {drift['r2']:.4f}")

    alert = abs(drift["slope"]) > DRIFT_ALERT_SLOPE
    if alert:
        total_hours = drift["trend_hours"][-1] - drift["trend_hours"][0]
        total_drift = drift["slope"] * total_hours
        print(f"  CLEANING ALERT: drift exceeds {DRIFT_ALERT_SLOPE} ADC/hr threshold")
        print(f"  Total drift: {total_drift:+.0f} ADC over {total_hours:.0f} hours")
    else:
        print(f"  Drift within tolerance (threshold: {DRIFT_ALERT_SLOPE} ADC/hr)")

    print("Generating plot...")
    plot(rows, all_windows, plateau_windows, drift, OUTPUT_PATH)


if __name__ == "__main__":
    main()
