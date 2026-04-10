# Sensor Drift Study

Detection of biofouling-induced sensor drift in DFRobot SEN0189 turbidity sensor data from a UK river deployment (Oct 2024).

## Key Findings

- **Dataset:** 50,338 cleaned readings across 4 sessions over 20 days (6-second sample interval)
- **Drift detected:** +1.01 ADC/hour over the 43-hour high-turbidity plateau (Session 4)
- **Total drift:** +43 ADC units, rising from ~830 to ~870+ ADC
- **R²:** 0.213 (moderate — consistent with gradual biofouling superimposed on natural variance)
- **Cleaning alert triggered:** Drift slope exceeds 0.5 ADC/hr threshold

The upward ADC drift under stable environmental conditions (temperature 11.3–12.3°C, TDS 11–26 ppm) is consistent with biofilm accumulation on the optical window, progressively attenuating the infrared signal.

## Method

1. Cleaned raw CSV: fixed header mismatch, removed temperature spike artifacts, tagged 4 recording sessions
2. Isolated the 43-hour high-turbidity plateau (Oct 26 14:00 – Oct 28 10:00) where environmental conditions were stable
3. Computed 30-minute rolling windows with 50% overlap
4. Filtered for "stable" windows (temperature stdev < 0.3°C, TDS stdev < 15 ppm, turbidity stdev < 60 ADC)
5. Fitted linear trend to stable-window medians

## Files

| File | Description |
|------|-------------|
| `drift_detection.py` | Experiment script (run with `python drift_detection.py`) |
| `drift_detection.png` | 3-panel output: full session, plateau drift trend, environmental context |
| `data01_clean.csv` | Cleaned input data (50,338 rows, 4 sessions) |

## Output

![Drift Detection](drift_detection.png)

- **Panel 1:** Full 3-day session — raw ADC scatter + rolling median, with plateau region highlighted
- **Panel 2:** Drift analysis zoomed to plateau — stable windows (green), linear trend (red), cleaning alert
- **Panel 3:** Environmental context — temperature and TDS confirm conditions were stable (drift is not environmental)
