"""Tier 1 data layer: pull real F1 race laps + weather from FastF1, clean them, save a CSV.

Run:  uv run fetch_data.py
"""

from pathlib import Path

import fastf1
import pandas as pd

HERE = Path(__file__).parent
CACHE_DIR = HERE / ".fastf1_cache"
DATA_DIR = HERE / "data"
OUT_CSV = DATA_DIR / "laps_clean.csv"

# Add a race = add a line here. (year, event) — event is what fastf1.get_session accepts.
RACES = [
    (2024, "Bahrain"),      # dry, hot track
    (2024, "Spain"),        # dry
    (2024, "Silverstone"),  # wet/mixed (doc: 0.427 rainfall fraction)
    (2024, "Canada"),       # wet/mixed (doc: 0.256)
    (2023, "Monza"),        # dry, low-degradation
    (2024, "Monza"),        # dry
    (2024, "Suzuka"),       # dry
    (2024, "Spielberg"),    # dry (Austrian GP)
    (2024, "Zandvoort"),    # dry (Dutch GP)
]
# Use circuit/location names above, not country names: fastf1's fuzzy matcher silently
# "corrects" e.g. 'Great Britain' to the Austrian GP. The summary prints what it resolved to.

# Columns worth keeping downstream. Anything missing gets reported, not silently dropped.
LAP_COLS = [
    "Driver", "DriverNumber", "Team", "LapNumber", "Stint", "LapTime",
    "Compound", "TyreLife", "FreshTyre", "TrackStatus", "IsAccurate", "LapStartTime",
]
WEATHER_COLS = ["AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed", "WindDirection", "Rainfall"]


def load_race(year, event):
    """Return one race's laps with weather merged on, plus (raw, after_f1, after_f2, after_f3) counts."""
    session = fastf1.get_session(year, event, "R")
    session.load(telemetry=False, messages=False)

    laps = session.laps.copy()
    raw = len(laps)

    missing = [c for c in LAP_COLS if c not in laps.columns]
    if missing:
        print(f"  !! missing lap columns {missing}")

    # Weather is sampled ~once/minute, laps are ~once/90s — nearest-time merge, not a join.
    weather = session.weather_data[["Time"] + WEATHER_COLS].sort_values("Time")
    laps = laps.sort_values("LapStartTime")
    laps = pd.merge_asof(laps, weather, left_on="LapStartTime", right_on="Time", direction="nearest")

    # Filter 1: FastF1's own lap-integrity flag.
    laps = laps[laps["IsAccurate"]]
    after_f1 = len(laps)

    # Filter 2: green flag only (drops safety car / VSC laps).
    laps = laps[laps["TrackStatus"] == "1"]
    after_f2 = len(laps)

    # Filter 3: under 1.05x this race's median — drops traffic, in-laps and out-laps.
    median = laps["LapTime"].median()
    laps = laps[laps["LapTime"] < 1.05 * median]
    after_f3 = len(laps)

    laps = laps[[c for c in LAP_COLS if c in laps.columns] + WEATHER_COLS].copy()
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    resolved = session.event["EventName"]
    laps.insert(0, "Event", resolved)
    laps.insert(0, "Year", year)
    return laps, (resolved, raw, after_f1, after_f2, after_f3)


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)  # FastF1 refuses a cache dir that doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    frames, stats = [], []
    for year, event in RACES:
        print(f"\n=== {year} {event} ===")
        laps, counts = load_race(year, event)
        frames.append(laps)
        stats.append((year, *counts))

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)

    # --- self-check summary: a dead race or a missing column has to be visible here ---
    print("\n" + "=" * 78)
    print(f"{'race (as resolved)':<32}{'raw':>7}{'IsAcc':>8}{'green':>8}{'<1.05x':>8}{'survived':>10}")
    print("-" * 78)
    total = [0, 0, 0, 0]
    for year, event, raw, f1_, f2, f3 in stats:
        flag = "  <-- ZERO LAPS" if f3 == 0 else ""
        print(f"{f'{year} {event}':<32}{raw:>7}{f1_:>8}{f2:>8}{f3:>8}{f3 / raw:>9.0%}{flag}")
        total = [a + b for a, b in zip(total, (raw, f1_, f2, f3))]
    print("-" * 78)
    raw, f1_, f2, f3 = total
    print(f"{'TOTAL':<32}{raw:>7}{f1_:>8}{f2:>8}{f3:>8}{f3 / raw:>9.0%}")
    print(f"\nfilters 1+2 kept {f2 / raw:.1%} of raw laps (doc expects roughly 85-90%)")
    print(f"final: {len(combined)} rows x {len(combined.columns)} cols -> {OUT_CSV}")
    print("compound counts:\n", combined["Compound"].value_counts().to_string())

    needed = ["Compound", "TyreLife", "LapTimeSeconds", "Driver", "Year", "Event"]
    assert not combined.empty, "no laps survived across all races"
    assert all(c in combined.columns for c in needed), f"missing required columns: {needed}"
    assert combined.groupby(["Year", "Event"]).size().min() > 0, "a race contributed zero laps"


if __name__ == "__main__":
    main()
