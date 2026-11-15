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

# Since Oct 2026 FastF1 (>= 3.9) needs an F1TV login for timing data,
# so only races already in .fastf1_cache are usable. The other 2023
# races and all of 2025 are blocked until a login is configured.

RACES = [
    (2022, "Bahrain"),
    (2022, "Spain"),
    (2022, "Silverstone"),
    (2022, "Canada"),
    (2022, "Monza"),
    (2022, "Suzuka"),
    # (2022, "Spielberg") excluded: broken source tyre data. 250 of 1,324 laps
    # have no compound and 298 no tyre age, and FastF1 warns it had to fix
    # incorrect tyre stint info for this race. Tyre data is what we model.
    (2022, "Zandvoort"),  # Dutch GP
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


# What fastf1.get_session must resolve each event key to.
# FastF1's fuzzy matching once silently loaded the wrong race.
EXPECTED_EVENT_NAMES = {
    "Bahrain": "Bahrain Grand Prix",
    "Spain": "Spanish Grand Prix",
    "Silverstone": "British Grand Prix",
    "Canada": "Canadian Grand Prix",
    "Monza": "Italian Grand Prix",
    "Suzuka": "Japanese Grand Prix",
    "Spielberg": "Austrian Grand Prix",
    "Zandvoort": "Dutch Grand Prix",
}


# =========================================================
# PIRELLI COMPOUND NOMINATIONS
# =========================================================

# FastF1 only gives weekend-relative labels (SOFT/MEDIUM/HARD).
# This maps them to the real Pirelli compound for each race.
# Covers more races than RACES so blocked races can be added later.

PIRELLI_NOMINATIONS = {
    (2022, "Bahrain"):     {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},  # https://www.formula1.com/en/latest/article/what-tyres-will-the-teams-and-drivers-have-for-the-2022-bahrain-grand-prix.5nQGRrYUMcq8Y2BQeHBTeK
    (2022, "Spain"):       {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},  # https://racingnews365.com/pirelli-confirm-tyre-compound-choices-for-upcoming-races
    (2022, "Silverstone"): {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},  # https://www.formula1.com/en/latest/article/what-tyres-will-the-teams-and-drivers-have-for-the-2022-british-grand-prix.5NSQn3DvP84qUkQIqltoxd
    (2022, "Canada"):      {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},  # https://www.formula1.com/en/latest/article/what-tyres-will-the-teams-and-drivers-have-for-the-2022-canadian-grand-prix.3fi2Bwiw5SHQIpsLC5iMNm
    (2022, "Monza"):       {"HARD": "C2", "MEDIUM": "C3", "SOFT": "C4"},  # https://press.pirelli.com/2022-italian-grand-prix---preview-2/
    (2022, "Suzuka"):      {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},  # https://racingnews365.com/pirelli-confirm-tyre-choices-for-return-of-singapore-and-japan
    (2022, "Spielberg"):   {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},  # https://www.pitpass.com/72881/Pirelli-reveals-compounds-for-Austria-France-and-Hungary
    (2022, "Zandvoort"):   {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},  # https://press.pirelli.com/2022-dutch-grand-prix--preview/
    (2023, "Bahrain"):     {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},  # https://www.formula1.com/en/latest/article/pirelli-confirm-tyre-choices-for-first-three-f1-races-of-2023-as-new.1xlpcTVaLqw0NeM3aYoyCt
    (2023, "Spain"):       {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},  # https://press.pirelli.com/2023-tyre-compound-choices--monaco-and-spain/
    (2023, "Silverstone"): {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},  # https://press.pirelli.com/2023-tyre-compound-choices--canada-austria-and-great-britain/
    (2023, "Canada"):      {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},  # https://press.pirelli.com/2023-tyre-compound-choices--canada-austria-and-great-britain/
    (2023, "Monza"):       {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},  # https://press.pirelli.com/news-and-tyre-choices-for-zandvoort-and-monza/
    (2023, "Suzuka"):      {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},  # https://press.pirelli.com/unchanged-nominations-for-singapore-japan-and-qatar--in-name-at-least/
    (2023, "Spielberg"):   {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},  # https://press.pirelli.com/2023-tyre-compound-choices--canada-austria-and-great-britain/
    (2023, "Zandvoort"):   {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},  # https://press.pirelli.com/news-and-tyre-choices-for-zandvoort-and-monza/
    (2024, "Bahrain"):     {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},  # https://press.pirelli.com/pirelli-nominates-the-compounds-for-the-start-of-2024/
    (2024, "Spain"):       {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},  # https://www.formula1.com/en/latest/article/what-tyres-will-the-teams-and-drivers-have-for-the-2024-spanish-grand-prix.7rLCqBvroeRw7HVW8GGChP
    (2024, "Silverstone"): {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},  # https://www.formula1.com/en/latest/article/what-tyres-will-the-teams-and-drivers-have-for-the-2024-british-grand-prix.3vzK0q1kbEOjzwGClOCXoF
    (2024, "Canada"):      {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},  # https://tyrepress.com/2024/04/pirelli-sticks-with-softest-f1-tyre-compound-trio-for-three-more-grands-prix
    (2024, "Monza"):       {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},  # https://www.formula1.com/en/latest/article/what-tyres-will-the-teams-and-drivers-have-for-the-2024-italian-grand-prix.2oO1jaVl2vB25F16qcFMPH
    (2024, "Suzuka"):      {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},  # https://www.formula1.com/en/latest/article/what-tyres-will-the-teams-and-drivers-have-for-the-2024-japanese-grand-prix.0dan38YDF6LLdAHyNuhJX
    (2024, "Spielberg"):   {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},  # https://www.formula1.com/en/latest/article/what-tyres-will-the-teams-and-drivers-have-for-the-2024-austrian-grand-prix.501nkNqLI40inoC4Tdf4w6
    (2024, "Zandvoort"):   {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},  # https://press.pirelli.com/all-compounds-on-track-over-next-three-races/
    (2025, "Bahrain"):     {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},  # https://www.formula1.com/en/latest/article/what-tyres-will-the-teams-and-drivers-have-for-the-2025-bahrain-grand-prix.7gnHbNpm0OJQIjPqsWoyof
    (2025, "Spain"):       {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},  # https://press.pirelli.com/from-monaco-to-montreal-all-the-compounds-in-play/
    (2025, "Silverstone"): {"HARD": "C2", "MEDIUM": "C3", "SOFT": "C4"},  # https://press.pirelli.com/both-new-and-familiar-for-spielberg-to-budapest/
    (2025, "Canada"):      {"HARD": "C4", "MEDIUM": "C5", "SOFT": "C6"},  # https://www.formula1.com/en/latest/article/what-tyres-will-the-teams-and-drivers-have-for-the-2025-canadian-grand-prix.6f3pKs15OHHlxe7uI544kO
    (2025, "Monza"):       {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},  # https://www.formula1.com/en/latest/article/what-tyres-will-the-teams-and-drivers-have-for-the-2025-italian-grand-prix.7fO9Z1g1aG6QUGhXzg8gLU
    (2025, "Suzuka"):      {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},  # https://scuderiafans.com/pirelli-reveals-tire-compounds-for-first-five-races-of-2025-f1-season/
    (2025, "Spielberg"):   {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},  # https://press.pirelli.com/both-new-and-familiar-for-spielberg-to-budapest/
    (2025, "Zandvoort"):   {"HARD": "C2", "MEDIUM": "C3", "SOFT": "C4"},  # https://www.pitpass.com/80783/Pirelli-reveals-compounds-for-remaining-races
}


def compound_code(year, event, compound):
    """
    Return a Pirelli code comparable across years ("C0".."C6"),
    or None for INTERMEDIATE/WET or an unknown race.
    """

    code = PIRELLI_NOMINATIONS.get((year, event), {}).get(compound)

    if code is None:
        return None

    # In 2023 Pirelli renamed the old C1 to C0 (a new C1 sits between them).
    if year == 2022 and code == "C1":
        return "C0"

    # ponytail: 2025 also changed compounds; no 2025 race is in RACES yet,
    # so 2025 codes may need their own mapping when those races are added.
    return code


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

    # Laps without a compound or tyre age can't be used by any tyre model.
    laps = laps.dropna(subset=["Compound", "TyreLife"])

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

    expected = EXPECTED_EVENT_NAMES[event]

    if resolved != expected:
        raise ValueError(
            f"FastF1 resolved {year} {event!r} to {resolved!r}, expected {expected!r}"
        )

    laps.insert(0, "Event", resolved)

    laps.insert(0, "Year", year)

    laps["CCode"] = [
        compound_code(year, event, compound) for compound in laps["Compound"]
    ]

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

    missing_nominations = [race for race in RACES if race not in PIRELLI_NOMINATIONS]

    assert not missing_nominations, (
        f"RACES without a PIRELLI_NOMINATIONS entry: {missing_nominations}"
    )

    assert compound_code(2022, "Bahrain", "HARD") == "C0"  # old C1 renamed
    assert compound_code(2024, "Bahrain", "HARD") == "C1"
    assert compound_code(2024, "Bahrain", "INTERMEDIATE") is None

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

    needed = [
        "Compound",
        "CCode",
        "TyreLife",
        "LapTimeSeconds",
        "Driver",
        "Year",
        "Event",
    ]

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
