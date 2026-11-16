"""
F1 TYRE STRATEGY — TIF1 DATA COLLECTOR

Training seasons: 2022–2025
Testing season: 2026
Data source: tif1 0.11.3
Output: One Parquet file per successfully processed race.
"""

from pathlib import Path
import logging
import json
import re
import time

import pandas as pd
import tif1


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

SEASONS = [2022, 2023, 2024, 2025]

MAX_RACES_PER_RUN = 90
DELAY_BETWEEN_RACES = 0

# Separate progress file for the new tif1 environment.
PROGRESS_FILE = BASE_DIR / "tif1_0113_progress.json"

# Persistent race data.
DATA_DIR = BASE_DIR / "training_data_tif1"
DATA_DIR.mkdir(exist_ok=True)


logging.getLogger("tif1").setLevel(logging.WARNING)


# =========================================================
# REQUIRED COLUMNS
# =========================================================

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
# DATA CLEANING
# =========================================================


def clean_race_laps(laps, year, round_number, event):

    laps = laps.copy()

    missing = [column for column in LAP_COLS if column not in laps.columns]

    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    for column in WEATHER_COLS:
        if column not in laps.columns:
            laps[column] = pd.NA

    raw_count = len(laps)

    # Accurate laps only.
    laps = laps[laps["IsAccurate"] == True].copy()
    after_accuracy = len(laps)

    # Green flag only.
    laps = laps[laps["TrackStatus"].astype(str) == "1"].copy()

    after_green = len(laps)

    if laps.empty:
        raise ValueError("No laps survived accuracy and track-status filters.")

    # Convert lap time to seconds.
    if "LapTimeSeconds" in laps.columns:
        laps["LapTimeSeconds"] = pd.to_numeric(
            laps["LapTimeSeconds"],
            errors="coerce",
        )
    else:
        laps["LapTimeSeconds"] = pd.to_timedelta(
            laps["LapTime"], errors="coerce"
        ).dt.total_seconds()

    laps = laps.dropna(subset=["LapTimeSeconds", "Compound", "TyreLife"]).copy()

    if laps.empty:
        raise ValueError("No laps with valid lap time and tyre data.")

    # Remove extremely slow laps.
    median = laps["LapTimeSeconds"].median()

    laps = laps[laps["LapTimeSeconds"] < 1.05 * median].copy()

    # Normalize compound names.
    laps["Compound"] = laps["Compound"].astype(str).str.upper()

    # Keep dry compounds.
    laps = laps[laps["Compound"].isin(["HARD", "MEDIUM", "SOFT"])].copy()

    if laps.empty:
        raise ValueError("No usable HARD, MEDIUM or SOFT laps.")

    keep = list(dict.fromkeys(LAP_COLS + WEATHER_COLS + ["LapTimeSeconds"]))

    laps = laps[keep].copy()

    laps.insert(0, "Event", event)
    laps.insert(0, "Round", round_number)
    laps.insert(0, "Year", year)

    stats = {
        "year": year,
        "round": round_number,
        "event": event,
        "raw": raw_count,
        "accurate": after_accuracy,
        "green": after_green,
        "usable": len(laps),
    }

    return laps, stats


# =========================================================
# LOAD ONE RACE
# =========================================================


def load_race(year, round_number, expected_event):

    # Use the official calendar round number.
    session = tif1.get_session(
        year,
        int(round_number),
        "Race",
    )

    laps = session.laps.copy()

    if laps.empty:
        raise ValueError("No lap data returned.")

    # tif1 0.11.3 uses nested session metadata.
    info = session.session_info

    if not isinstance(info, dict):
        raise ValueError("Invalid session metadata.")

    meeting = info.get("Meeting", {})

    if not isinstance(meeting, dict):
        raise ValueError("Missing meeting metadata.")

    resolved_event = meeting.get("Name")

    if not resolved_event:
        raise ValueError("Could not verify returned event name.")

    resolved_event = str(resolved_event).strip()
    expected_event = str(expected_event).strip()

    if resolved_event.casefold() != expected_event.casefold():
        raise ValueError(
            f"Event mismatch: expected '{expected_event}', received '{resolved_event}'."
        )

    cleaned, stats = clean_race_laps(
        laps,
        year,
        round_number,
        resolved_event,
    )

    return cleaned, stats


# =========================================================
# EVENT SCHEDULES
# =========================================================


def get_race_candidates():

    candidates = []

    for year in SEASONS:
        print(f"\nReading {year} event schedule...")

        schedule = tif1.get_events(year)

        if not isinstance(schedule, pd.DataFrame):
            schedule = pd.DataFrame(schedule)

        required = ["RoundNumber", "EventName"]

        missing = [column for column in required if column not in schedule.columns]

        if missing:
            raise ValueError(f"{year} schedule missing columns: {missing}")

        schedule = schedule.copy()

        schedule["RoundNumber"] = pd.to_numeric(
            schedule["RoundNumber"],
            errors="coerce",
        )

        schedule = schedule[schedule["RoundNumber"] > 0]

        for _, row in schedule.iterrows():
            candidates.append(
                (
                    year,
                    int(row["RoundNumber"]),
                    str(row["EventName"]).strip(),
                )
            )

        print(f"Race candidates: {len(schedule)}")

    return candidates


# =========================================================
# PROGRESS TRACKING
# =========================================================


def load_completed_races():

    if not PROGRESS_FILE.exists():
        return set()

    try:
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))

        return set(data.get("completed", []))

    except (json.JSONDecodeError, OSError):
        return set()


def save_completed_races(completed):

    data = {"completed": sorted(completed)}

    temp_file = PROGRESS_FILE.with_suffix(".tmp")

    temp_file.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    temp_file.replace(PROGRESS_FILE)


# =========================================================
# PARQUET STORAGE
# =========================================================


def get_output_path(year, round_number, event):

    slug = (
        re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            event,
        )
        .strip("_")
        .lower()
    )

    filename = f"{year}_R{round_number:02d}_{slug}.parquet"

    return DATA_DIR / filename


def save_race_data(laps, year, round_number, event):

    output_path = get_output_path(
        year,
        round_number,
        event,
    )

    laps.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
    )

    # Verify that the saved file can be read.
    verification = pd.read_parquet(output_path)

    if len(verification) != len(laps):
        raise ValueError("Parquet verification failed.")

    return output_path


# =========================================================
# LOAD TRAINING BATCH
# =========================================================


def load_training_data():

    candidates = get_race_candidates()
    completed = load_completed_races()

    pending = [
        (year, round_number, event)
        for year, round_number, event in candidates
        if f"{year}|{event}" not in completed
    ]

    print("\n" + "=" * 70)
    print("TIF1 0.11.3 TRAINING DATA COLLECTION")
    print("=" * 70)

    print(f"Total candidates: {len(candidates)}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(pending)}")

    selected = pending[:MAX_RACES_PER_RUN]

    frames = []
    stats = []
    unavailable = []

    for index, (year, round_number, event) in enumerate(
        selected,
        start=1,
    ):
        print(
            f"\n[{index}/{len(selected)}] {year} — Round {round_number}: {event}",
            flush=True,
        )

        try:
            laps, race_stats = load_race(
                year,
                round_number,
                event,
            )

            output_path = save_race_data(
                laps,
                year,
                round_number,
                event,
            )

            # Mark completed only after successful saving.
            completed.add(f"{year}|{event}")
            save_completed_races(completed)

            frames.append(laps)
            stats.append(race_stats)

            print(f"[PASS] {len(laps):,} usable laps")
            print(f"[SAVED] {output_path.name}")

        except Exception as error:
            unavailable.append((year, event, str(error)))

            print(f"[FAILED] {error}")

        if index < len(selected):
            time.sleep(DELAY_BETWEEN_RACES)

    print(f"\nProgress: {len(completed)}/{len(candidates)} races completed.")

    if not frames:
        print("\nNo new race data was collected.")
        return None, stats, unavailable

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = combined.sort_values(
        ["Year", "Round", "Driver", "LapNumber"]
    ).reset_index(drop=True)

    return combined, stats, unavailable


# =========================================================
# SUMMARY
# =========================================================


def print_summary(combined, stats, unavailable):

    print("\n" + "=" * 70)
    print("F1 TYRE STRATEGY — DATA COLLECTION SUMMARY")
    print("=" * 70)

    print(f"{'YEAR':<8}{'RACE':<35}{'USABLE LAPS':>15}")
    print("-" * 70)

    for item in stats:
        print(f"{item['year']:<8}{item['event']:<35}{item['usable']:>15,}")

    print("-" * 70)

    print(f"Races loaded: {len(stats)}")
    print(f"Races failed: {len(unavailable)}")
    print(f"Total usable laps: {len(combined):,}")

    print("\nCOMPOUND COUNTS")
    print(combined["Compound"].value_counts().to_string())

    print("\nWEATHER COVERAGE")

    for column in ["Rainfall", "TrackTemp"]:
        missing = combined[column].isna().sum()

        print(f"{column}: {len(combined) - missing:,} available | {missing:,} missing")

    if unavailable:
        print("\nFAILED RACES")

        for year, event, reason in unavailable:
            print(f"{year} — {event}: {reason}")

    print("\n" + "=" * 70)
    print(f"Parquet files stored in: {DATA_DIR}")
    print("No CSV or plot files created.")
    print("=" * 70)


# =========================================================
# MAIN
# =========================================================


def main():

    combined, stats, unavailable = load_training_data()

    if combined is None:
        print("No new data collected in this batch.")
        return None

    print_summary(
        combined,
        stats,
        unavailable,
    )

    return combined


if __name__ == "__main__":
    TRAINING_DATA = main()
