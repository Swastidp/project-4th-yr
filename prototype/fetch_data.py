"""
Tier 1 data layer: pull real F1 race laps + weather from FastF1,
clean them, and save a CSV.

Run:
    uv run fetch_data.py
"""

from pathlib import Path

import fastf1
import pandas as pd


# =========================================================
# PATHS
# =========================================================

HERE = Path(__file__).parent

CACHE_DIR = HERE / ".fastf1_cache"
DATA_DIR = HERE / "data"

OUT_CSV = DATA_DIR / "laps_clean.csv"


# =========================================================
# RACES
# =========================================================

# Add a race = add a line here.
# Format: (year, event)
# Event is what fastf1.get_session accepts.

RACES = [
    (2024, "Bahrain"),  # dry, hot track
    (2024, "Spain"),  # dry
    (2024, "Silverstone"),  # wet/mixed
    (2024, "Canada"),  # wet/mixed
    (2023, "Monza"),  # dry, low-degradation
    (2024, "Monza"),  # dry
    (2024, "Suzuka"),  # dry
    (2024, "Spielberg"),  # dry (Austrian GP)
    (2024, "Zandvoort"),  # dry (Dutch GP)
]


# =========================================================
# COLUMNS
# =========================================================

# Lap-level columns worth keeping downstream.
LAP_COLS = [
    "Driver",
    "DriverNumber",
    "Team",
    "LapNumber",
    "Stint",
    "LapTime",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "TrackStatus",
    "IsAccurate",
    "LapStartTime",
]

# Weather columns worth keeping downstream.
WEATHER_COLS = [
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "WindDirection",
    "Rainfall",
]


# =========================================================
# LOAD ONE RACE
# =========================================================


def load_race(year, event):
    """
    Return one race's cleaned laps with weather merged on.

    Also returns:
        (resolved_event, raw, after_f1, after_f2, after_f3)
    """

    # -----------------------------------------------------
    # Load FastF1 race session
    # -----------------------------------------------------

    session = fastf1.get_session(year, event, "R")

    session.load(telemetry=False, messages=False)

    laps = session.laps.copy()

    raw = len(laps)

    # -----------------------------------------------------
    # Check required lap columns
    # -----------------------------------------------------

    missing_lap = [column for column in LAP_COLS if column not in laps.columns]

    if missing_lap:
        print(f"  !! missing lap columns: {missing_lap}")

    # -----------------------------------------------------
    # Check required weather columns
    # -----------------------------------------------------

    weather_source = session.weather_data.copy()

    missing_weather = [
        column for column in WEATHER_COLS if column not in weather_source.columns
    ]

    if missing_weather:
        print(f"  !! missing weather columns: {missing_weather}")

    # -----------------------------------------------------
    # Keep only available weather columns
    # -----------------------------------------------------

    available_weather = [
        column for column in WEATHER_COLS if column in weather_source.columns
    ]

    weather = weather_source[["Time"] + available_weather].sort_values("Time")

    # -----------------------------------------------------
    # Sort laps by lap start time
    # -----------------------------------------------------

    laps = laps.sort_values("LapStartTime")

    # -----------------------------------------------------
    # Merge weather using nearest timestamp
    #
    # Weather is sampled approximately once per minute,
    # while laps occur roughly once every 90 seconds.
    # Therefore this is a nearest-time merge.
    # -----------------------------------------------------

    laps = pd.merge_asof(
        laps, weather, left_on="LapStartTime", right_on="Time", direction="nearest"
    )

    # -----------------------------------------------------
    # Filter 1:
    # FastF1's own lap-integrity flag
    # -----------------------------------------------------

    laps = laps[laps["IsAccurate"]]

    after_f1 = len(laps)

    # -----------------------------------------------------
    # Filter 2:
    # Green flag only
    #
    # Drops Safety Car / VSC laps.
    # -----------------------------------------------------

    laps = laps[laps["TrackStatus"] == "1"]

    after_f2 = len(laps)

    # -----------------------------------------------------
    # Filter 3:
    # Remove unusually slow laps
    #
    # Keep laps below 1.05 × race median.
    # This reduces traffic, in-lap and out-lap effects.
    # -----------------------------------------------------

    median = laps["LapTime"].median()

    laps = laps[laps["LapTime"] < 1.05 * median]

    after_f3 = len(laps)

    # -----------------------------------------------------
    # Keep required columns that actually exist
    # -----------------------------------------------------

    available_lap_cols = [column for column in LAP_COLS if column in laps.columns]

    laps = laps[available_lap_cols + available_weather].copy()

    # -----------------------------------------------------
    # Convert lap time to seconds
    # -----------------------------------------------------

    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

    # -----------------------------------------------------
    # Use the event name resolved by FastF1
    # -----------------------------------------------------

    resolved = session.event["EventName"]

    laps.insert(0, "Event", resolved)

    laps.insert(0, "Year", year)

    return (laps, (resolved, raw, after_f1, after_f2, after_f3))


# =========================================================
# MAIN
# =========================================================


def main():

    # -----------------------------------------------------
    # Create required directories
    # -----------------------------------------------------

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Enable FastF1 cache
    # -----------------------------------------------------

    fastf1.Cache.enable_cache(str(CACHE_DIR))

    frames = []
    stats = []

    # -----------------------------------------------------
    # Process every race
    # -----------------------------------------------------

    for year, event in RACES:
        print(f"\n=== {year} {event} ===")

        laps, counts = load_race(year, event)

        frames.append(laps)

        stats.append((year, *counts))

    # -----------------------------------------------------
    # Combine all races
    # -----------------------------------------------------

    if not frames:
        raise RuntimeError("No race data was loaded.")

    combined = pd.concat(frames, ignore_index=True)

    # -----------------------------------------------------
    # Save cleaned dataset
    # -----------------------------------------------------

    combined.to_csv(OUT_CSV, index=False)

    # =====================================================
    # SELF-CHECK SUMMARY
    # =====================================================

    print("\n" + "=" * 78)

    print(
        f"{'race (as resolved)':<32}"
        f"{'raw':>7}"
        f"{'IsAcc':>8}"
        f"{'green':>8}"
        f"{'<1.05x':>8}"
        f"{'survived':>10}"
    )

    print("-" * 78)

    total = [0, 0, 0, 0]

    for year, event, raw, f1_count, f2, f3 in stats:
        flag = "  <-- ZERO LAPS" if f3 == 0 else ""

        survival_rate = f3 / raw if raw > 0 else 0

        print(
            f"{f'{year} {event}':<32}"
            f"{raw:>7}"
            f"{f1_count:>8}"
            f"{f2:>8}"
            f"{f3:>8}"
            f"{survival_rate:>9.0%}"
            f"{flag}"
        )

        total = [a + b for a, b in zip(total, (raw, f1_count, f2, f3))]

    print("-" * 78)

    raw, f1_count, f2, f3 = total

    total_survival = f3 / raw if raw > 0 else 0

    print(f"{'TOTAL':<32}{raw:>7}{f1_count:>8}{f2:>8}{f3:>8}{total_survival:>9.0%}")

    # -----------------------------------------------------
    # Filter summary
    # -----------------------------------------------------

    filter12_rate = f2 / raw if raw > 0 else 0

    print(
        f"\nfilters 1+2 kept "
        f"{filter12_rate:.1%} of raw laps "
        f"(doc expects roughly 85-90%)"
    )

    print(f"final: {len(combined)} rows x {len(combined.columns)} cols -> {OUT_CSV}")

    # -----------------------------------------------------
    # Compound counts
    # -----------------------------------------------------

    print("compound counts:\n", combined["Compound"].value_counts().to_string())

    # =====================================================
    # VALIDATION
    # =====================================================

    needed = ["Compound", "TyreLife", "LapTimeSeconds", "Driver", "Year", "Event"]

    assert not combined.empty, "no laps survived across all races"

    missing_required = [column for column in needed if column not in combined.columns]

    assert not missing_required, f"missing required columns: {missing_required}"

    race_sizes = combined.groupby(["Year", "Event"]).size()

    assert not race_sizes.empty and race_sizes.min() > 0, "a race contributed zero laps"


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
