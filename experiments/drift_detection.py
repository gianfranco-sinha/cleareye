"""Drift detection experiment on cleaned Session 4 data.

Analyses the 3-day continuous recording (Session 4, Oct 25–28) to detect
sensor fouling.  Strategy:

1. Compute rolling median of turbidity_adc.
2. Detect anomalous jumps — sudden large shifts in rolling median that
   indicate environmental events, not gradual fouling.
3. Segment the data into stable regimes between jumps.
4. Within each segment, identify stable environmental windows (temp/TDS
   variance low) and fit a linear drift trend.
5. Output a multi-panel graph showing the full session, detected anomalies,
   per-segment drift trends, and environmental context.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data01_clean.csv"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "output" / "drift_detection.png"

WINDOW_MINUTES = 30          # rolling window size
TEMP_STABILITY_STD = 0.3     # max temp stdev within window to count as "stable"
TDS_STABILITY_STD = 15.0     # max TDS stdev within window
TURB_STABILITY_STD = 60.0    # max turbidity stdev within window (reject transients)
DRIFT_ALERT_SLOPE = 0.5      # ADC units / hour — threshold for cleaning alert
SESSION = "4"

# Anomaly detection: a jump in hourly median exceeding this threshold
# is classified as an anomalous event, not gradual drift.
JUMP_THRESHOLD_ADC = 150     # ADC units change between consecutive hourly medians
JUMP_COOLDOWN_HOURS = 2      # min hours of stability after a jump before a segment starts


@dataclass
class Segment:
    """A contiguous stable-regime slice of the data."""
    label: str
    start: datetime
    end: datetime
    rows: list[dict]
    windows: list[dict]
    drift: dict


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


def hourly_medians(rows: list[dict]) -> list[dict]:
    """Compute hourly median turbidity for jump detection."""
    if not rows:
        return []
    start = rows[0]["ts"]
    end = rows[-1]["ts"]
    cursor = start
    results = []
    while cursor + timedelta(hours=1) <= end:
        subset = [r["turb"] for r in rows if cursor <= r["ts"] < cursor + timedelta(hours=1)]
        if subset:
            results.append({
                "ts": cursor + timedelta(minutes=30),
                "median": float(np.median(subset)),
                "start": cursor,
                "end": cursor + timedelta(hours=1),
            })
        cursor += timedelta(hours=1)
    return results


def detect_jumps(hourly: list[dict]) -> list[dict]:
    """Find anomalous jumps in hourly medians."""
    jumps = []
    for i in range(1, len(hourly)):
        delta = hourly[i]["median"] - hourly[i - 1]["median"]
        if abs(delta) > JUMP_THRESHOLD_ADC:
            jumps.append({
                "ts": hourly[i]["ts"],
                "delta": delta,
                "from_level": hourly[i - 1]["median"],
                "to_level": hourly[i]["median"],
            })
    return jumps


def segment_data(rows: list[dict], jumps: list[dict]) -> list[Segment]:
    """Split data into stable segments between anomalous jumps.

    Applies a cooldown after each jump to let the signal settle before
    starting the next segment.
    """
    cooldown = timedelta(hours=JUMP_COOLDOWN_HOURS)

    # Build boundary list: session start, after each jump+cooldown, session end
    boundaries = [rows[0]["ts"]]
    for j in jumps:
        # End current segment just before the jump
        boundaries.append(j["ts"] - timedelta(minutes=30))
        # Start next segment after cooldown
        boundaries.append(j["ts"] + cooldown)
    boundaries.append(rows[-1]["ts"])

    segments = []
    for i in range(0, len(boundaries) - 1, 2):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]
        seg_rows = [r for r in rows if seg_start <= r["ts"] <= seg_end]
        if len(seg_rows) < 100:
            continue
        label = chr(ord("A") + len(segments))
        segments.append(Segment(
            label=label,
            start=seg_start,
            end=seg_end,
            rows=seg_rows,
            windows=[],
            drift={},
        ))
    return segments


def rolling_windows(rows: list[dict], window_min: int) -> list[dict]:
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
    """Fit linear trend to stable-window turbidity medians."""
    stable = [
        w for w in windows
        if w["temp_std"] < TEMP_STABILITY_STD
        and w["tds_std"] < TDS_STABILITY_STD
        and w["turb_std"] < TURB_STABILITY_STD
    ]
    if len(stable) < 3:
        return {
            "slope": 0.0, "r2": 0.0, "stable_windows": stable,
            "trend_hours": np.array([]), "trend_vals": np.array([]),
        }

    t0 = stable[0]["ts_center"]
    hours = np.array([(w["ts_center"] - t0).total_seconds() / 3600 for w in stable])
    medians = np.array([w["turb_median"] for w in stable])

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


SEGMENT_COLORS = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]


def plot(
    rows: list[dict],
    all_windows: list[dict],
    jumps: list[dict],
    segments: list[Segment],
    output: Path,
) -> None:
    """Generate a 4-panel drift detection figure."""
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    fig.suptitle(
        "Sensor Drift Detection with Anomaly Exclusion — Session 4 (Oct 25–28 2024)",
        fontsize=14, fontweight="bold",
    )

    ts_all = [r["ts"] for r in rows]
    turb_all = [r["turb"] for r in rows]

    # ── Panel 1: Full session with anomalies marked ──────────────────
    ax1 = axes[0]
    ax1.scatter(ts_all, turb_all, s=0.3, alpha=0.15, color="steelblue", label="Raw ADC")
    win_ts = [w["ts_center"] for w in all_windows]
    win_med = [w["turb_median"] for w in all_windows]
    ax1.plot(win_ts, win_med, color="darkorange", linewidth=1.5, label=f"Rolling median ({WINDOW_MINUTES}min)")

    for j in jumps:
        ax1.axvline(j["ts"], color="red", linewidth=2, linestyle="--", alpha=0.8)
        ax1.annotate(
            f"Anomaly\n{j['delta']:+.0f} ADC",
            xy=(j["ts"], max(j["from_level"], j["to_level"])),
            xytext=(10, 20), textcoords="offset points",
            fontsize=8, color="red", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red"),
        )

    # Shade segments
    for i, seg in enumerate(segments):
        color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
        ax1.axvspan(seg.start, seg.end, alpha=0.06, color=color)

    ax1.set_ylabel("Turbidity ADC")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Full Session — Anomalous Jumps Detected (red dashed)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Per-segment drift trends ────────────────────────────
    ax2 = axes[1]
    legend_handles = []
    for i, seg in enumerate(segments):
        color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
        stable = seg.drift.get("stable_windows", [])
        if not stable:
            continue

        stable_ts = [w["ts_center"] for w in stable]
        stable_med = [w["turb_median"] for w in stable]
        ax2.scatter(stable_ts, stable_med, s=10, color=color, alpha=0.6)

        trend_vals = seg.drift.get("trend_vals", np.array([]))
        slope = seg.drift.get("slope", 0.0)
        r2 = seg.drift.get("r2", 0.0)
        if len(trend_vals) > 0:
            ax2.plot(stable_ts, trend_vals, color=color, linewidth=2.5)

        hours = seg.drift["trend_hours"]
        total_h = hours[-1] - hours[0] if len(hours) > 1 else 0
        alert = abs(slope) > DRIFT_ALERT_SLOPE
        marker = " *" if alert else ""
        legend_handles.append(mpatches.Patch(
            color=color,
            label=f"Seg {seg.label}: {slope:+.2f} ADC/hr, R²={r2:.3f}, {total_h:.0f}h{marker}",
        ))

    # Overall alert check
    any_alert = any(abs(seg.drift.get("slope", 0)) > DRIFT_ALERT_SLOPE for seg in segments)
    alert_text = "CLEANING ALERT" if any_alert else "Within tolerance"
    alert_color = "red" if any_alert else "green"
    ax2.text(
        0.98, 0.05, alert_text, transform=ax2.transAxes,
        fontsize=12, fontweight="bold", color=alert_color,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=alert_color, alpha=0.9),
    )

    ax2.set_ylabel("Turbidity ADC (stable windows)")
    ax2.legend(handles=legend_handles, loc="upper left", fontsize=9)
    ax2.set_title("Per-Segment Drift Trends (anomalies excluded, * = alert)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Zoomed view of the longest segment ──────────────────
    ax3 = axes[2]
    longest = max(segments, key=lambda s: len(s.rows))
    seg_rows_ts = [r["ts"] for r in longest.rows]
    seg_rows_turb = [r["turb"] for r in longest.rows]
    ax3.scatter(seg_rows_ts, seg_rows_turb, s=0.3, alpha=0.1, color="steelblue")

    # Overlay windows and trend
    all_seg_win_ts = [w["ts_center"] for w in longest.windows]
    all_seg_win_med = [w["turb_median"] for w in longest.windows]
    ax3.plot(all_seg_win_ts, all_seg_win_med, color="darkorange", linewidth=1, alpha=0.5, label="Rolling median")

    stable = longest.drift.get("stable_windows", [])
    if stable:
        st_ts = [w["ts_center"] for w in stable]
        st_med = [w["turb_median"] for w in stable]
        ax3.scatter(st_ts, st_med, s=14, color="seagreen", zorder=3, label="Stable windows")

        trend_vals = longest.drift.get("trend_vals", np.array([]))
        slope = longest.drift["slope"]
        r2 = longest.drift["r2"]
        hours = longest.drift["trend_hours"]
        if len(trend_vals) > 0:
            ax3.plot(st_ts, trend_vals, color="red", linewidth=2.5,
                     label=f"Drift: {slope:+.2f} ADC/hr (R²={r2:.3f})")
            total_h = hours[-1] - hours[0]
            total_drift = slope * total_h
            ax3.text(
                0.98, 0.18, f"Total drift: {total_drift:+.0f} ADC over {total_h:.0f}h",
                transform=ax3.transAxes, fontsize=10, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.9),
            )

    ax3.set_ylabel("Turbidity ADC")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.set_title(f"Segment {longest.label} Detail ({longest.start.strftime('%b %d %H:%M')} → {longest.end.strftime('%b %d %H:%M')})")
    ax3.set_xlim(longest.start, longest.end)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Environmental context ───────────────────────────────
    ax4 = axes[3]
    ax4.scatter(ts_all, [r["temp"] for r in rows], s=0.3, alpha=0.15, color="coral", label="Temperature (°C)")
    ax4_twin = ax4.twinx()
    ax4_twin.scatter(ts_all, [r["tds"] for r in rows], s=0.3, alpha=0.15, color="mediumpurple", label="TDS (ppm)")
    ax4.set_ylabel("Temperature (°C)", color="coral")
    ax4_twin.set_ylabel("TDS (ppm)", color="mediumpurple")
    ax4.set_xlabel("Time")

    for j in jumps:
        ax4.axvline(j["ts"], color="red", linewidth=1.5, linestyle="--", alpha=0.5)

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax4.set_title("Environmental Context (anomaly boundaries shown)")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    ax4.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax4.grid(True, alpha=0.3)

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


def main() -> None:
    print("Loading Session 4 data...")
    rows = load_session(DATA_PATH, SESSION)
    print(f"  {len(rows):,} rows loaded")

    print("\nComputing hourly medians for jump detection...")
    hourly = hourly_medians(rows)

    print("Detecting anomalous jumps...")
    jumps = detect_jumps(hourly)
    for j in jumps:
        print(f"  {j['ts'].strftime('%b %d %H:%M')}: {j['delta']:+.0f} ADC "
              f"({j['from_level']:.0f} → {j['to_level']:.0f})")
    if not jumps:
        print("  (none detected)")

    print(f"\nSegmenting data ({JUMP_COOLDOWN_HOURS}h cooldown after each jump)...")
    segments = segment_data(rows, jumps)
    for seg in segments:
        duration_h = (seg.end - seg.start).total_seconds() / 3600
        print(f"  Segment {seg.label}: {seg.start.strftime('%b %d %H:%M')} → "
              f"{seg.end.strftime('%b %d %H:%M')} ({duration_h:.1f}h, {len(seg.rows):,} rows)")

    print("\nComputing rolling windows (full session)...")
    all_windows = rolling_windows(rows, WINDOW_MINUTES)

    print("Analysing drift per segment...")
    for seg in segments:
        seg.windows = rolling_windows(seg.rows, WINDOW_MINUTES)
        seg.drift = detect_drift(seg.windows)
        slope = seg.drift["slope"]
        r2 = seg.drift["r2"]
        n_stable = len(seg.drift["stable_windows"])
        n_total = len(seg.windows)
        hours = seg.drift["trend_hours"]
        total_h = hours[-1] - hours[0] if len(hours) > 1 else 0

        alert = " ** CLEANING ALERT **" if abs(slope) > DRIFT_ALERT_SLOPE else ""
        print(f"  Segment {seg.label}: {slope:+.3f} ADC/hr, R²={r2:.4f}, "
              f"{n_stable}/{n_total} stable windows, {total_h:.0f}h span{alert}")
        if total_h > 0:
            print(f"    Total drift: {slope * total_h:+.0f} ADC over {total_h:.0f}h")

    print("\nGenerating plot...")
    plot(rows, all_windows, jumps, segments, OUTPUT_PATH)


if __name__ == "__main__":
    main()
