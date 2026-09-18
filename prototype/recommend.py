"""
Stage-3 compound + stint-length recommender.

No new model is trained here.

The recommender reuses the Stage-1 Ridge pace-loss model and
the Stage-2 Kaplan-Meier stint-life estimates.

For a dry race:

    Cost(c, L) =
        cumulative predicted pace loss
        + pit-stop loss * (remaining laps / stint length)

The lowest-cost option is ranked first.

Important limitations:
    - PIT_LOSS_S is a domain assumption, not learned from the data.
    - Track temperature is not used in the recommendation cost.
    - Circuit-specific compound effects cannot be reliably separated
      with the current small 9-race dataset.
    - Rainfall currently routes to INTERMEDIATE by rule rather than
      through a trained wet-tyre model.

Run:
    uv run recommend.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from build_stints import km_table
from train_pace_loss import (
    IN_CSV,
    add_features,
    make_ridge,
    ridge_design,
)


# =========================================================
# PATHS / SETTINGS
# =========================================================

STINTS_CSV = Path(__file__).parent / "data" / "stints.csv"

DRY = [
    "HARD",
    "MEDIUM",
    "SOFT",
]

MAX_STINT = 40

# Assumed pit-lane / pit-stop time loss.
# This is a domain assumption, not a learned model parameter.
PIT_LOSS_S = 22.0

# Warn when the requested stint is much longer than
# the compound's Kaplan-Meier median.
KM_SLACK = 1.5

TOP_N = 8


# =========================================================
# LOAD LAP DATA
# =========================================================


def load_laps():
    """
    Load the same cleaned lap dataset used by Stage 1
    and recreate Delta/FuelPct using the same feature function.
    """

    if not IN_CSV.exists():
        raise FileNotFoundError(f"{IN_CSV} not found. Run fetch_data.py first.")

    df = add_features(pd.read_csv(IN_CSV))

    if df.empty:
        raise RuntimeError(f"{IN_CSV} is empty.")

    required = [
        "Compound",
        "Delta",
        "TyreLife",
        "FuelPct",
        "TrackTemp",
        "Year",
        "Event",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


# =========================================================
# RIDGE MODEL
# =========================================================


def fit_ridge(df=None):
    """
    Refit the Stage-1 Ridge model on the complete
    available cleaned lap dataset.

    This uses exactly the same basic Ridge design as
    train_pace_loss.py.

    TrackTemp remains part of the fitted Stage-1 model,
    but it is deliberately held at zero in the
    recommendation prediction design so the recommender
    does not claim to be track-temperature aware.
    """

    if df is None:
        df = load_laps()

    compounds = sorted(df["Compound"].dropna().unique().tolist())

    if not compounds:
        raise RuntimeError("No compounds available for Ridge fitting.")

    X = ridge_design(
        df,
        compounds,
    )

    y = df["Delta"].astype(float)

    # TrackTemp is deliberately removed from the
    # recommendation design. It is not used by the
    # cost function.
    X = X.drop(columns=["TrackTemp"])

    bad = X.columns[X.isna().any()].tolist()

    if bad:
        raise ValueError(f"NaNs leaked into the recommendation design matrix: {bad}")

    model = make_ridge()

    model.fit(
        X,
        y,
    )

    return (
        model,
        compounds,
    )


# =========================================================
# RECOMMENDATION DESIGN
# =========================================================


def recommendation_design(
    df,
    compounds,
):
    """
    Build the Ridge design used by the recommender.

    This mirrors ridge_design() from Stage 1 but removes
    TrackTemp because the recommendation layer intentionally
    does not adjust its cost for track temperature.
    """

    X = ridge_design(
        df.assign(TrackTemp=0.0),
        compounds,
    )

    return X.drop(columns=["TrackTemp"])


# =========================================================
# KAPLAN-MEIER MEDIANS
# =========================================================


def km_medians():
    """
    Get Kaplan-Meier median stint lengths for the
    dry compounds from Stage 2.
    """

    if not STINTS_CSV.exists():
        raise FileNotFoundError(f"{STINTS_CSV} not found. Run build_stints.py first.")

    stints = pd.read_csv(STINTS_CSV)

    required = [
        "stint_length",
        "event",
        "Compound",
    ]

    missing = [column for column in required if column not in stints.columns]

    if missing:
        raise ValueError(f"Missing required stint columns: {missing}")

    available = [
        compound for compound in DRY if compound in set(stints["Compound"].dropna())
    ]

    if not available:
        raise RuntimeError("No dry compounds found in stints.csv.")

    km, _ = km_table(
        stints,
        available,
    )

    return dict(
        zip(
            km["compound"],
            km["km_median"],
        )
    )


# =========================================================
# FITTED PACE
# =========================================================


def fitted(
    model,
    compounds,
    compound,
    ages,
    fuel,
):
    """
    Predict pace loss for one compound over a sequence
    of tyre ages.

    TrackTemp is deliberately excluded from the
    recommendation cost.
    """

    ages = np.asarray(
        ages,
        dtype=float,
    )

    fuel = np.asarray(
        fuel,
        dtype=float,
    )

    if fuel.ndim == 0:
        fuel = np.full(
            len(ages),
            float(fuel),
        )

    grid = pd.DataFrame(
        {
            "TyreLife": ages,
            "Compound": compound,
            "FuelPct": fuel,
            "TrackTemp": 0.0,
        }
    )

    X = recommendation_design(
        grid,
        compounds,
    )

    return model.predict(X)


# =========================================================
# CUMULATIVE PACE-LOSS COST
# =========================================================


def cumulative_loss(
    model,
    compounds,
    compound,
    current_lap,
    total_race_laps,
    max_stint,
):
    """
    Calculate cumulative predicted pace loss for
    stint lengths from 1 to max_stint.

    Element L-1 represents:

        sum of predicted pace loss
        for tyre ages 1 ... L.
    """

    ages = np.arange(
        1,
        max_stint + 1,
        dtype=float,
    )

    future_laps = np.minimum(
        current_lap + ages,
        total_race_laps,
    )

    fuel = 1.0 - future_laps / total_race_laps

    predictions = fitted(
        model,
        compounds,
        compound,
        ages,
        fuel,
    )

    return np.cumsum(predictions)


# =========================================================
# COST TABLE
# =========================================================


def cost_table(
    model,
    compounds,
    medians,
    current_lap,
    total_race_laps,
):
    """
    Calculate every compound/stint-length combination.

    R = remaining race laps

    Cost =
        cumulative predicted pace-loss burden
        + PIT_LOSS_S * R / L
    """

    if total_race_laps <= current_lap:
        raise ValueError("current_lap must be less than total_race_laps.")

    remaining_laps = total_race_laps - current_lap

    max_stint = int(
        min(
            remaining_laps,
            MAX_STINT,
        )
    )

    if max_stint < 1:
        raise ValueError("No laps remain for recommendation.")

    rows = []

    for compound in DRY:
        if compound not in compounds:
            continue

        cumulative = cumulative_loss(
            model,
            compounds,
            compound,
            current_lap,
            total_race_laps,
            max_stint,
        )

        median = medians.get(
            compound,
            np.nan,
        )

        for stint_laps in range(
            1,
            max_stint + 1,
        ):
            degradation_cost = cumulative[stint_laps - 1]

            pit_cost = PIT_LOSS_S * remaining_laps / stint_laps

            total_cost = degradation_cost + pit_cost

            if pd.notna(median):
                if stint_laps > KM_SLACK * median:
                    flag = f"> {KM_SLACK}x KM median ({median:.0f})"
                else:
                    flag = ""
            else:
                flag = "KM median unavailable"

            rows.append(
                {
                    "compound": compound,
                    "stint_laps": stint_laps,
                    "deg_cost": degradation_cost,
                    "pit_cost": pit_cost,
                    "cost": total_cost,
                    "km_median": median,
                    "flag": flag,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "compound",
                "stint_laps",
                "deg_cost",
                "pit_cost",
                "cost",
                "km_median",
                "flag",
            ]
        )

    return pd.DataFrame(rows).sort_values("cost").reset_index(drop=True)


# =========================================================
# RECOMMEND
# =========================================================


def recommend(
    current_lap,
    total_race_laps,
    rainfall,
    compounds_available=DRY,
    _fit=None,
    _medians=None,
):
    """
    Return ranked dry compound/stint-length options.

    If rainfall is detected, use the current simple
    wet-condition rule and route to INTERMEDIATE.

    No wet-tyre ML model is claimed here.
    """

    if rainfall:
        return pd.DataFrame(
            [
                {
                    "compound": "INTERMEDIATE",
                    "stint_laps": np.nan,
                    "deg_cost": np.nan,
                    "pit_cost": np.nan,
                    "cost": np.nan,
                    "km_median": np.nan,
                    "flag": (
                        "by rule: rainfall detected; dry cost optimisation skipped"
                    ),
                }
            ]
        )

    if _fit is None:
        df = load_laps()
        fit = fit_ridge(df)
    else:
        fit = _fit

    model, compounds = fit

    if _medians is None:
        medians = km_medians()
    else:
        medians = _medians

    available = [
        compound
        for compound in compounds_available
        if compound in DRY and compound in compounds
    ]

    if not available:
        raise ValueError("No valid dry compounds are available for recommendation.")

    table = cost_table(
        model,
        compounds,
        medians,
        current_lap,
        total_race_laps,
    )

    table = table[table["compound"].isin(available)].copy()

    return table.sort_values("cost").reset_index(drop=True)


# =========================================================
# DISPLAY
# =========================================================


def show(
    title,
    table,
):
    """
    Print a readable recommendation table.
    """

    print(f"\n--- {title} ---")

    if table.empty:
        print("No valid recommendation.")
        return

    if table["cost"].isna().all():
        row = table.iloc[0]

        print(f"{row['compound']} ({row['flag']})")

        return

    print(
        f"{'#':<4}"
        f"{'compound':<10}"
        f"{'stint L':>9}"
        f"{'deg cost':>10}"
        f"{'pit cost':>10}"
        f"{'TOTAL s':>10}"
        "   flag"
    )

    for index, row in table.head(TOP_N).iterrows():
        print(
            f"{index + 1:<4}"
            f"{row['compound']:<10}"
            f"{row['stint_laps']:>9.0f}"
            f"{row['deg_cost']:>10.1f}"
            f"{row['pit_cost']:>10.1f}"
            f"{row['cost']:>10.1f}"
            f"   {row['flag']}"
        )

    best = table.loc[table.groupby("compound")["cost"].idxmin()].sort_values("cost")

    print(
        "  best L per compound: "
        + ",  ".join(
            f"{row['compound']} L={row['stint_laps']:.0f} ({row['cost']:.1f}s)"
            for _, row in best.iterrows()
        )
    )


# =========================================================
# COMPOUND COEFFICIENTS
# =========================================================


def compound_coefs(
    df,
    fit,
):
    """
    Return the fitted compound intercept, age coefficient,
    age^2 coefficient and average fitted gradient.

    This is diagnostic information only.
    """

    model, compounds = fit

    X = recommendation_design(
        df,
        compounds,
    )

    coef = pd.Series(
        model.coef_,
        index=X.columns,
    )

    rows = {}

    for compound in compounds:
        ages = pd.to_numeric(
            df.loc[
                df["Compound"] == compound,
                "TyreLife",
            ],
            errors="coerce",
        ).dropna()

        if ages.empty:
            rows[compound] = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )
            continue

        hi = ages.quantile(0.98)

        b1 = coef[f"age[{compound}]"]

        b2 = coef[f"age2[{compound}]"]

        if hi <= 1:
            net = np.nan
        else:
            net = ((b1 * hi + b2 * hi**2 / 100.0) - (b1 + b2 / 100.0)) / (hi - 1)

        rows[compound] = (
            coef[f"int[{compound}]"],
            b1,
            b2,
            net,
        )

    return rows


def print_coef_table(
    coefficients,
    compounds,
):
    """
    Print fitted compound coefficients.
    """

    print("\n--- per-compound Ridge coefficients ---")

    print(
        f"{'compound':<14}"
        f"{'intercept':>12}"
        f"{'age s/lap':>12}"
        f"{'age^2/100':>12}"
        f"{'net s/lap':>12}"
    )

    for compound in compounds:
        values = coefficients[compound]

        print(
            f"{compound:<14}"
            f"{values[0]:>12.3f}"
            f"{values[1]:>12.4f}"
            f"{values[2]:>12.4f}"
            f"{values[3]:>12.4f}"
        )

    print(
        "net s/lap = average gradient of the fitted "
        "curve over the compound's observed age range"
    )


# =========================================================
# MAIN
# =========================================================


def main():

    # -----------------------------------------------------
    # Load and fit
    # -----------------------------------------------------

    df = load_laps()

    fit = fit_ridge(df)

    model, compounds = fit

    medians = km_medians()

    # -----------------------------------------------------
    # Basic information
    # -----------------------------------------------------

    print(f"{len(df)} laps, {df['Race'].nunique()} races")

    print("compounds: " + ", ".join(compounds))

    print(f"\npit loss assumption = {PIT_LOSS_S:.0f}s")

    print("cost function = cumulative predicted pace-loss burden + pit-stop cost")

    print("TrackTemp is not used by the recommendation cost.")

    print(
        "circuit-specific compound effects are "
        "not separately estimated because the "
        "current dataset is too small for reliable "
        "compound × circuit interactions."
    )

    print(
        "\nKM median stint lengths: "
        + ", ".join(
            f"{compound} {median:.0f}"
            for compound, median in medians.items()
            if pd.notna(median)
        )
    )

    # -----------------------------------------------------
    # Coefficients
    # -----------------------------------------------------

    coefficients = compound_coefs(
        df,
        fit,
    )

    print_coef_table(
        coefficients,
        compounds,
    )

    # -----------------------------------------------------
    # Demonstration scenarios
    # -----------------------------------------------------

    early = recommend(
        current_lap=5,
        total_race_laps=57,
        rainfall=False,
        _fit=fit,
        _medians=medians,
    )

    late = recommend(
        current_lap=45,
        total_race_laps=57,
        rainfall=False,
        _fit=fit,
        _medians=medians,
    )

    wet = recommend(
        current_lap=20,
        total_race_laps=57,
        rainfall=True,
        _fit=fit,
        _medians=medians,
    )

    show(
        "Scenario 1: lap 5 of 57, R = 52 laps left",
        early,
    )

    show(
        "Scenario 2: lap 45 of 57, R = 12 laps left",
        late,
    )

    show(
        "Scenario 3: lap 20 of 57, rainfall = True",
        wet,
    )

    # -----------------------------------------------------
    # Pace sanity information
    # -----------------------------------------------------

    fuel = df["FuelPct"].mean()

    print("\n--- fitted pace loss at tyre age = 1 ---")

    print(f"{'compound':<14}{'predicted delta':>18}")

    for compound in DRY:
        if compound not in compounds:
            continue

        prediction = fitted(
            model,
            compounds,
            compound,
            np.array([1.0]),
            fuel,
        )[0]

        print(f"{compound:<14}{prediction:>18.3f}")

    print(
        "\nNote: the fitted age-1 values are diagnostic "
        "model outputs. They are not treated as a required "
        "Soft > Medium > Hard validation ordering."
    )

    # -----------------------------------------------------
    # Self-checks
    # -----------------------------------------------------

    # Deliberately create a long remaining race to verify
    # that the KM warning can fire at long stint lengths.
    extreme = cost_table(
        model,
        compounds,
        medians,
        current_lap=5,
        total_race_laps=200,
    )

    long_flags = sorted(
        extreme.loc[
            (extreme["stint_laps"] == MAX_STINT) & (extreme["flag"] != ""),
            "compound",
        ].tolist()
    )

    short_flags = (
        extreme.loc[
            extreme["stint_laps"] == 5,
            "flag",
        ]
        == ""
    ).all()

    checks = [
        (
            "Ridge model fitted successfully",
            model is not None,
        ),
        (
            "recommendation table is non-empty",
            not early.empty and not late.empty,
        ),
        (
            "all dry recommendation costs are finite",
            np.isfinite(early["cost"]).all() and np.isfinite(late["cost"]).all(),
        ),
        (
            "all dry recommendation costs are positive",
            (early["cost"] > 0).all() and (late["cost"] > 0).all(),
        ),
        (
            "rainfall routes to INTERMEDIATE",
            (not wet.empty and wet.iloc[0]["compound"] == "INTERMEDIATE"),
        ),
        (
            "rainfall skips dry cost optimisation",
            wet["cost"].isna().all(),
        ),
        (
            "long-stint KM warning can fire",
            bool(long_flags),
        ),
        (
            "short-stint KM warning stays silent",
            bool(short_flags),
        ),
    ]

    print("\n" + "=" * 62)

    for label, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")

    print("=" * 62)

    passed = sum(ok for _, ok in checks)

    print(f"{passed}/{len(checks)} checks passed")

    print("\nLimitations:")

    print(f"  - PIT_LOSS_S = {PIT_LOSS_S:.0f}s is an assumed value.")

    print("  - Track temperature is not used by the recommendation cost.")

    print("  - Wet conditions use a simple rainfall -> INTERMEDIATE rule.")

    print(
        "  - Compound × circuit effects require more race data for reliable estimation."
    )

    print(
        "  - KM median is an observed stint-survival "
        "statistic, not a direct tyre-degradation limit."
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
