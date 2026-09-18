"""
Stage-2 stint-life model: estimate how long a tyre stint remains in use
using survival analysis.

A stint may end because:
    - the driver changes tyres / begins another stint,
    - the race finishes,
    - the driver retires.

The last stint of a driver-race is therefore right-censored.

IMPORTANT:
The current event definition represents a transition to another stint.
It is not a direct physical measurement of tyre degradation.

This script deliberately does NOT read laps_clean.csv.
The quick-lap filters used for the pace model can remove laps around
pit stops, safety-car periods, or slow laps and would therefore shorten
observed stint lengths.

Instead, RAW FastF1 session laps are loaded from the local FastF1 cache.

Run:
    uv run build_stints.py
"""

import time
import warnings
from pathlib import Path
import numpy as np

import fastf1
import matplotlib

import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceError

from fetch_data import (
    CACHE_DIR,
    RACES,
    WEATHER_COLS,
)


# =========================================================
# PATHS / SETTINGS
# =========================================================

DATA_DIR = Path(__file__).parent / "data"

OUT_CSV = DATA_DIR / "stints.csv"
OUT_PNG = DATA_DIR / "survival_curves.png"

DRY = [
    "HARD",
    "MEDIUM",
    "SOFT",
]

MIN_STINTS_FOR_COX = 20

COX_COVARIATES = [
    "is_soft",
    "is_medium",
    "is_intermediate",
    "start_lap",
    "total_race_laps",
    "Humidity",
    "TrackTemp",
    "starting_tyre_age",
]


# =========================================================
# BUILD STINT TABLE
# =========================================================


def stints_for_race(year, event):
    """
    Load raw FastF1 race data and build one row for each
    driver tyre stint.

    The function also adds race weather and metadata
    needed for the survival analysis.
    """

    # Load the race session from the FastF1 cache.
    session = fastf1.get_session(year, event, "R")

    # Load race laps without telemetry or message data.
    session.load(
        telemetry=False,
        messages=False,
    )

    # Copy the raw lap data for processing.
    laps = session.laps.copy()

    # Stop if no raw lap data is available.
    if laps.empty:
        raise RuntimeError(f"No raw laps available for {year} {event}.")

    # Determine the total number of laps completed or scheduled.
    observed_laps = laps["LapNumber"].max()

    scheduled_laps = getattr(session, "total_laps", 0) or 0

    total_race_laps = int(
        max(
            observed_laps,
            scheduled_laps,
        )
    )

    # Sort laps chronologically before grouping them into stints.
    laps = laps.sort_values("LapNumber")

    # Group laps by driver and FastF1 stint number.
    grouped = laps.groupby(
        ["Driver", "Stint"],
        dropna=True,
    )

    # Calculate the main characteristics of each tyre stint.
    stints = grouped.agg(
        stint_length=("LapNumber", "size"),
        start_lap=("LapNumber", "min"),
        end_lap=("LapNumber", "max"),
        starting_tyre_age=("TyreLife", "first"),
        start_time=("LapStartTime", "first"),
        Compound=("Compound", "first"),
        n_compounds=("Compound", "nunique"),
        pit_in=("PitInTime", lambda series: series.notna().sum()),
    ).reset_index()

    # Load weather information recorded during the race.
    weather_source = session.weather_data.copy()

    # Check which expected weather columns are available.
    missing_weather = [
        column for column in WEATHER_COLS if column not in weather_source.columns
    ]

    if missing_weather:
        print(f"  !! missing weather columns for {year} {event}: {missing_weather}")

    # Keep only weather columns that actually exist.
    available_weather = [
        column for column in WEATHER_COLS if column in weather_source.columns
    ]

    # Weather data must contain timestamps for matching.
    if "Time" not in weather_source.columns:
        raise RuntimeError(f"Weather data has no Time column for {year} {event}.")

    weather = weather_source[["Time"] + available_weather].sort_values("Time")

    # Use the first weather timestamp for stints without a lap start time.
    stints["start_time"] = stints["start_time"].fillna(pd.Timedelta(0))

    # Match each stint with the nearest weather observation.
    stints = pd.merge_asof(
        stints.sort_values("start_time"),
        weather,
        left_on="start_time",
        right_on="Time",
        direction="nearest",
    )

    # Add missing weather columns as empty values.
    for column in WEATHER_COLS:
        if column not in stints.columns:
            stints[column] = pd.NA

    # Add the total race distance to every stint.
    stints["total_race_laps"] = total_race_laps

    # Store the official event name returned by FastF1.
    resolved_event = session.event["EventName"]

    # Add event and year information to the beginning of the table.
    stints.insert(0, "Event", resolved_event)
    stints.insert(0, "Year", year)

    return stints, resolved_event


# =========================================================
# CENSORING
# =========================================================


def add_censoring(stints):
    """
    Add survival-analysis event information.

    event = 1 means another stint followed this stint.
    event = 0 means this was the driver's final observed stint,
    so its exact endpoint is censored.
    """

    # Work on a copy so the original DataFrame is not modified.
    stints = stints.copy()

    # Identify each driver-race combination.
    key = [
        "Year",
        "Event",
        "Driver",
    ]

    # Find the final stint number for each driver in each race.
    max_stint = stints.groupby(key)["Stint"].transform("max")

    # Mark stints followed by another stint as observed events.
    stints["event"] = (stints["Stint"] < max_stint).astype(int)

    # Check whether the final stint reached the race distance.
    # One lap of tolerance accounts for lapped cars.
    stints["finished_race"] = stints["end_lap"] >= stints["total_race_laps"] - 1

    return stints


# =========================================================
# KAPLAN-MEIER
# =========================================================


def km_table(stints, compounds):
    """
    Calculate Kaplan-Meier survival statistics for
    each tyre compound.

    Returns the summary statistics and fitted KM models.
    """

    # Store the summary statistics for each compound.
    rows = []

    # Store the fitted Kaplan-Meier models.
    fitters = {}

    # Analyse each tyre compound separately.
    for compound in compounds:
        # Select stints belonging to the current compound.
        group = stints[stints["Compound"] == compound].copy()

        # Skip compounds with no available data.
        if group.empty:
            continue

        # Remove rows missing survival-analysis variables.
        group = group.dropna(
            subset=[
                "stint_length",
                "event",
            ]
        )

        if group.empty:
            continue

        # Create a Kaplan-Meier estimator.
        kmf = KaplanMeierFitter()

        # Fit the survival model using stint length
        # and the censoring event indicator.
        kmf.fit(
            group["stint_length"],
            group["event"],
            label=compound,
        )

        # Save the fitted model for later plotting/export.
        fitters[compound] = kmf

        # Store raw and Kaplan-Meier statistics.
        rows.append(
            {
                "compound": compound,
                "stints": len(group),
                "events": int(group["event"].sum()),
                "censored_pct": (100 * (1 - group["event"].mean())),
                "raw_mean": group["stint_length"].mean(),
                "raw_median": group["stint_length"].median(),
                "km_median": kmf.median_survival_time_,
            }
        )

    # Return both the summary table and fitted models.
    return pd.DataFrame(rows), fitters


# =========================================================
# EXPORT KM DATA
# =========================================================


def export_km_csv(fitters):
    """
    Export Kaplan-Meier survival probabilities
    and their 95% confidence intervals to CSV.
    """

    # Store survival data from all compounds.
    all_data = []

    # Process each fitted Kaplan-Meier model.
    for compound, kmf in fitters.items():
        # Extract the survival probability at each stint length.
        survival = kmf.survival_function_.iloc[:, 0]

        # Extract the 95% confidence interval.
        confidence = kmf.confidence_interval_

        # Create a table for the current compound.
        data = pd.DataFrame(
            {
                "StintLength": survival.index,
                compound: survival.values,
                f"{compound}_lower": confidence.iloc[:, 0].values,
                f"{compound}_upper": confidence.iloc[:, 1].values,
            }
        )

        # Use stint length as the table index.
        all_data.append(data.set_index("StintLength"))

    # Stop if no Kaplan-Meier models were generated.
    if not all_data:
        raise RuntimeError("No Kaplan-Meier curves were available for export.")

    # Combine all compounds into one table.
    km_data = pd.concat(
        all_data,
        axis=1,
    )

    # Sort by stint length and carry forward survival values.
    km_data = km_data.sort_index().ffill()

    # Set the final index name.
    km_data.index.name = "StintLength"

    # Convert the index back into a normal column.
    km_data = km_data.reset_index()

    # Save the processed Kaplan-Meier data.
    out_csv = DATA_DIR / "km_survival.csv"

    km_data.to_csv(
        out_csv,
        index=False,
    )

    print(f"saved {out_csv}")


# =========================================================
# KAPLAN-MEIER PLOT
# =========================================================


def make_plot(fitters):
    """
    Create and save the Kaplan-Meier survival plot
    for the available tyre compounds.
    """

    # Create the figure and plotting area.
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Define colours matching the F1 tyre compounds.
    compound_colors = {
        "SOFT": "#FF1801",
        "MEDIUM": "#FFD100",
        "HARD": "#FFFFFF",
        "INTERMEDIATE": "#00D26A",
        "WET": "#008CFF",
    }

    # Set the dark background for the chart.
    fig.patch.set_facecolor("#111113")
    ax.set_facecolor("#111113")

    # Plot the survival curve for each compound.
    for compound, kmf in fitters.items():
        # Use the predefined compound colour.
        color = compound_colors.get(
            compound,
            "#FFFFFF",
        )

        # Draw the Kaplan-Meier survival curve
        # together with its confidence interval.
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            color=color,
            linewidth=2.2,
        )

        # Make the confidence interval visually transparent.
        if ax.collections:
            ax.collections[-1].set_alpha(0.16)

    # Remove the automatically generated legend.
    legend = ax.get_legend()

    if legend is not None:
        legend.remove()

    # Add axis labels.
    ax.set_xlabel(
        "Stint length (laps)",
        color="#E8E8E8",
        fontsize=11,
    )

    ax.set_ylabel(
        "P(stint still running)",
        color="#E8E8E8",
        fontsize=11,
    )

    # Add the chart title.
    ax.set_title(
        "Kaplan–Meier Stint Survival by Compound",
        color="#FFFFFF",
        fontsize=16,
        pad=12,
    )

    # Style the chart axes.
    ax.tick_params(colors="#B8B8B8")

    for spine in ax.spines.values():
        spine.set_color("#3A3A3D")

    # Adjust spacing and save the final image.
    fig.tight_layout()

    fig.savefig(
        OUT_PNG,
        dpi=200,
        facecolor="#111113",
        bbox_inches="tight",
    )

    # Close the figure to release memory.
    plt.close(fig)

    print(f"\nsaved {OUT_PNG}")


# =========================================================
# MAIN
# =========================================================


def main():

    # -----------------------------------------------------
    # FastF1 cache
    # -----------------------------------------------------

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fastf1.Cache.enable_cache(str(CACHE_DIR))

    # Never silently download new data.
    # If something is missing, the user should rerun
    # fetch_data.py explicitly.
    fastf1.Cache.offline_mode(True)

    # -----------------------------------------------------
    # Load all races
    # -----------------------------------------------------

    frames = []

    start_time = time.time()

    for year, event in RACES:
        try:
            stint_frame, resolved = stints_for_race(
                year,
                event,
            )

        except Exception as exc:
            raise SystemExit(
                f"\nCACHE MISS on "
                f"{year} {event}: "
                f"{type(exc).__name__}: {exc}\n"
                "Offline mode is on, so nothing was "
                "re-downloaded.\n"
                "Run fetch_data.py to populate the "
                "FastF1 cache, then run build_stints.py again."
            ) from exc

        print(
            f"{year} "
            f"{resolved:<34} "
            f"{len(stint_frame):>4} stints, "
            f"{stint_frame['Driver'].nunique():>2} drivers"
        )

        frames.append(stint_frame)

    if not frames:
        raise RuntimeError("No race stint data was loaded.")

    print(
        f"loaded {len(RACES)} sessions "
        f"from cache in "
        f"{time.time() - start_time:.1f}s "
        "(no network)"
    )

    # -----------------------------------------------------
    # Combine and censor
    # -----------------------------------------------------

    stints = pd.concat(
        frames,
        ignore_index=True,
    )

    stints = add_censoring(stints)

    # -----------------------------------------------------
    # Data-quality validation
    # -----------------------------------------------------

    # A single stint should not contain multiple compounds.
    bad = stints[stints["n_compounds"] > 1]

    assert bad.empty, (
        f"{len(bad)} stints span more than one compound:\n{bad.head().to_string()}"
    )

    # Remove unknown compound labels.
    stints = stints.dropna(subset=["Compound"])

    stints = stints[
        ~stints["Compound"].isin(
            [
                "UNKNOWN",
                "TEST_UNKNOWN",
            ]
        )
    ].copy()

    assert not stints.empty, "No valid stints remain after compound filtering."

    # -----------------------------------------------------
    # Save stint dataset
    # -----------------------------------------------------

    stints.to_csv(
        OUT_CSV,
        index=False,
    )

    print(
        f"\n{len(stints)} stints, "
        f"{stints.groupby(['Year', 'Event']).ngroups} races, "
        f"{stints.groupby(['Year', 'Event', 'Driver']).ngroups} "
        f"driver-races -> {OUT_CSV}"
    )

    # -----------------------------------------------------
    # Censoring summary
    # -----------------------------------------------------

    censored = stints[stints["event"] == 0]

    event_fraction = stints["event"].mean()

    print(f"censored: {len(censored)} of {len(stints)} ({1 - event_fraction:.1%})")

    print(
        f"  of those, {int(censored['finished_race'].sum())} reached the race distance"
    )

    print(
        f"  {int((~censored['finished_race']).sum())} "
        "stopped early or were otherwise censored"
    )

    # -----------------------------------------------------
    # Data-quality information
    # -----------------------------------------------------

    short = stints[stints["stint_length"] <= 2]

    print("\ndata quality:")

    print(f"  {len(short)} stints of <=2 laps")

    suzuka_short = (short["Event"] == "Japanese Grand Prix").sum()

    print(f"  {suzuka_short} of them at Suzuka")

    pit_censored = censored[censored["pit_in"] > 0]

    print(f"  {len(pit_censored)} censored stints show a pit entry on their last lap")

    for compound in sorted(set(stints["Compound"]) - set(DRY)):
        by_race = (
            stints.loc[stints["Compound"] == compound]
            .groupby(["Year", "Event"])
            .size()
            .sort_values()
        )

        if by_race.empty:
            continue

        total = by_race.sum()

        print(
            f"  {compound}: "
            f"{total} stints from "
            f"{len(by_race)} race(s), "
            f"{by_race.iloc[-1] / total:.0%} "
            "from one race — interpret cautiously"
        )

    starting_age = pd.to_numeric(
        stints["starting_tyre_age"],
        errors="coerce",
    )

    print(
        f"  starting_tyre_age is 1-based: "
        f"{int((starting_age == 1).sum())} stints "
        "start fresh"
    )

    used_count = (starting_age > 1).sum()

    print(f"  {int(used_count)} stints start on an already-used set")

    # =====================================================
    # KAPLAN-MEIER
    # =====================================================

    compounds = [compound for compound in DRY if compound in set(stints["Compound"])]

    other_compounds = sorted(set(stints["Compound"]) - set(DRY))

    compounds += other_compounds

    km, fitters = km_table(
        stints,
        compounds,
    )

    if km.empty:
        raise RuntimeError("Kaplan-Meier analysis produced no results.")

    print("\n--- Kaplan-Meier vs raw stint length (laps) ---")

    print(
        f"{'compound':<14}"
        f"{'stints':>7}"
        f"{'events':>8}"
        f"{'cens %':>8}"
        f"{'raw mean':>10}"
        f"{'raw med':>9}"
        f"{'KM med':>8}"
        f"{'KM-raw':>8}"
    )

    for _, row in km.iterrows():
        if pd.isna(row["km_median"]):
            gap = np.nan
        else:
            gap = row["km_median"] - row["raw_median"]

        gap_text = f"{gap:+8.1f}" if pd.notna(gap) else f"{'NA':>8}"

        print(
            f"{row['compound']:<14}"
            f"{row['stints']:>7.0f}"
            f"{row['events']:>8.0f}"
            f"{row['censored_pct']:>7.0f}%"
            f"{row['raw_mean']:>10.1f}"
            f"{row['raw_median']:>9.1f}"
            f"{row['km_median']:>8.1f}"
            f"{gap_text}"
        )

    print(
        "KM estimates account for censored final stints; "
        "do not interpret the difference as direct physical "
        "tyre degradation."
    )

    # =====================================================
    # COX PROPORTIONAL HAZARDS MODEL
    # =====================================================

    counts = stints["Compound"].value_counts()

    keep = [
        compound
        for compound in compounds
        if counts[compound] >= MIN_STINTS_FOR_COX or compound == "HARD"
    ]

    dropped = [
        f"{compound} (n={counts[compound]})"
        for compound in compounds
        if compound not in keep
    ]

    cox_df = stints[stints["Compound"].isin(keep)].copy()

    # HARD is the baseline compound.
    for compound in [
        "SOFT",
        "MEDIUM",
        "INTERMEDIATE",
    ]:
        cox_df[f"is_{compound.lower()}"] = (cox_df["Compound"] == compound).astype(
            float
        )

    print("\n--- Cox PH (baseline HARD) ---")

    print(
        "excluded from the fit: "
        + (", ".join(dropped) if dropped else "nothing")
        + f" "
        f"(threshold: <{MIN_STINTS_FOR_COX} stints)"
    )

    # -----------------------------------------------------
    # Select usable covariates
    # -----------------------------------------------------

    usable_covariates = []

    for column in COX_COVARIATES:
        if column not in cox_df.columns:
            print(f"  skipping missing covariate: {column}")
            continue

        numeric = pd.to_numeric(
            cox_df[column],
            errors="coerce",
        )

        cox_df[column] = numeric

        if numeric.nunique(dropna=True) > 1:
            usable_covariates.append(column)

    if not usable_covariates:
        raise RuntimeError("No varying covariates are available for the Cox model.")

    cox_df = cox_df[
        usable_covariates
        + [
            "stint_length",
            "event",
        ]
    ].dropna()

    if cox_df.empty:
        raise RuntimeError("No complete rows remain for the Cox model.")

    # -----------------------------------------------------
    # Fit Cox model
    # -----------------------------------------------------

    cph = CoxPHFitter()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        try:
            cph.fit(
                cox_df,
                duration_col="stint_length",
                event_col="event",
            )

        except ConvergenceError as exc:
            raise SystemExit(f"Cox model did not converge: {exc}") from exc

    for warning in caught:
        print("  lifelines warning: " + str(warning.message).splitlines()[0][:110])

    # -----------------------------------------------------
    # Cox results
    # -----------------------------------------------------

    summary = cph.summary

    print(f"\n{'covariate':<22}{'coef':>9}{'hazard ratio':>14}{'p':>10}{'':>4}")

    for name, row in summary.iterrows():
        if row["p"] < 0.001:
            significance = "***"
        elif row["p"] < 0.01:
            significance = "**"
        elif row["p"] < 0.05:
            significance = "*"
        else:
            significance = ""

        print(
            f"{name:<22}"
            f"{row['coef']:>9.3f}"
            f"{row['exp(coef)']:>14.2f}"
            f"{row['p']:>10.3f}"
            f"{significance:>4}"
        )

    print(f"\nC-index {cph.concordance_index_:.3f} (0.5 = random ordering)")

    print(f"fitted on {len(cox_df)} stints, {int(cox_df['event'].sum())} events")

    # -----------------------------------------------------
    # Interpretation note
    # -----------------------------------------------------

    print(
        "\nCox interpretation: hazard ratios describe "
        "the association with the observed stint-transition "
        "event under this dataset's censoring definition. "
        "They are not direct measurements of physical tyre wear."
    )

    # =====================================================
    # EXPORTS
    # =====================================================

    export_km_csv(fitters)

    make_plot(fitters)

    # =====================================================
    # SUMMARY CHECKS
    # =====================================================

    print("\n" + "=" * 62)

    checks = [
        (
            f"stint table non-empty ({len(stints)} stints)",
            not stints.empty,
        ),
        (
            f"censoring detected, not degenerate ({event_fraction:.1%} events)",
            0 < event_fraction < 1,
        ),
        (
            "Cox model converged",
            hasattr(
                cph,
                "params_",
            )
            and cph.params_.notna().all(),
        ),
        (
            f"C-index in (0, 1): {cph.concordance_index_:.3f}",
            0 < cph.concordance_index_ < 1,
        ),
    ]

    for label, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")

    print("=" * 62)

    n_ok = sum(ok for _, ok in checks)

    print(
        f"{n_ok}/{len(checks)} checks passed"
        f"{'' if n_ok == len(checks) else '  <-- see FAIL lines above'}"
    )

    print(
        f"caveat: "
        f"{stints.groupby(['Year', 'Event']).ngroups} "
        "races only. Treat compound-specific survival "
        "differences cautiously because stint termination "
        "is influenced by race strategy and other factors."
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
