"""
Prepare Kaplan-Meier data for web visualisation.

Input:
    data/stints.csv

Output:
    data/kaplan_meier_web.csv

Output columns:
    Time  -> observed stint length
    Event -> 1 = stint transition observed, 0 = right-censored
    Group -> tyre compound

Run:
    uv run prepare_km_web.py
"""

from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

DATA_DIR = Path(__file__).parent / "data"

INPUT_CSV = DATA_DIR / "stints.csv"
OUTPUT_CSV = DATA_DIR / "kaplan_meier_web.csv"


# =========================================================
# MAIN
# =========================================================


def main():

    # -----------------------------------------------------
    # Check input
    # -----------------------------------------------------

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"{INPUT_CSV} not found. Run build_stints.py first.")

    # -----------------------------------------------------
    # Load stint data
    # -----------------------------------------------------

    df = pd.read_csv(INPUT_CSV)

    if df.empty:
        raise RuntimeError(f"{INPUT_CSV} is empty.")

    # -----------------------------------------------------
    # Check required columns
    # -----------------------------------------------------

    required_columns = [
        "stint_length",
        "event",
        "Compound",
    ]

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # -----------------------------------------------------
    # Keep only columns required by the web KM chart
    # -----------------------------------------------------

    km = df[required_columns].copy()

    # -----------------------------------------------------
    # Rename to standard web visualisation names
    # -----------------------------------------------------

    km = km.rename(
        columns={
            "stint_length": "Time",
            "event": "Event",
            "Compound": "Group",
        }
    )

    # -----------------------------------------------------
    # Convert numeric columns
    # -----------------------------------------------------

    km["Time"] = pd.to_numeric(
        km["Time"],
        errors="coerce",
    )

    km["Event"] = pd.to_numeric(
        km["Event"],
        errors="coerce",
    )

    # -----------------------------------------------------
    # Remove incomplete observations
    # -----------------------------------------------------

    km = km.dropna(
        subset=[
            "Time",
            "Event",
            "Group",
        ]
    ).copy()

    if km.empty:
        raise RuntimeError(
            "No valid Kaplan-Meier observations remain after removing missing values."
        )

    # -----------------------------------------------------
    # Validate event indicator
    # -----------------------------------------------------

    km["Event"] = km["Event"].astype(int)

    invalid_events = sorted(set(km["Event"]) - {0, 1})

    if invalid_events:
        raise ValueError(f"Event must contain only 0 or 1. Found: {invalid_events}")

    # -----------------------------------------------------
    # Validate time
    # -----------------------------------------------------

    if (km["Time"] <= 0).any():
        invalid_count = int((km["Time"] <= 0).sum())

        raise ValueError(
            f"{invalid_count} observations have non-positive stint length."
        )

    # -----------------------------------------------------
    # Clean group labels
    # -----------------------------------------------------

    km["Group"] = km["Group"].astype(str).str.strip()

    km = km[km["Group"] != ""].copy()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    km.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print(f"Saved: {OUTPUT_CSV}")

    print(f"Rows: {len(km)}")

    print("\nColumns:")

    print(km.columns.tolist())

    print("\nGroups:")

    print(km["Group"].value_counts().to_string())

    print("\nEvent counts:")

    print(km["Event"].value_counts().sort_index().to_string())


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
