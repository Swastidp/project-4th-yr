"""
Stage-1 pace-loss model: predict lap delta vs the driver's own best lap
in that race.

Two models on the same target:
    - Ridge with an explicit per-compound specification, for interpretation.
    - XGBoost, for predictive accuracy.

Both are evaluated using GroupKFold by race (Year + Event), so
no race appears in both training and testing.

FuelPct is used to separate fuel-burn effects from tyre-age effects.

Run:
    uv run train_pace_loss.py
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor


# =========================================================
# PATHS / SETTINGS
# =========================================================

DATA_DIR = Path(__file__).parent / "data"

IN_CSV = DATA_DIR / "laps_clean.csv"
OUT_PNG = DATA_DIR / "degradation_curve_corrected.png"

MIN_LAPS = 20

XGB_FEATURES = [
    "TyreLife",
    "FuelPct",
    "TrackTemp",
    "AirTemp",
    "Humidity",
]

# A race is treated as a separate wet regime when rainfall
# is present on more than 10% of its clean laps.
WET_FRACTION = 0.10


# =========================================================
# FEATURE ENGINEERING
# =========================================================


def add_features(df):
    """
    Create the modelling target and fuel-load correction.

    Delta:
        Lap time minus the driver's own best lap in that race.

    FuelPct:
        1 - LapNumber / race total laps.

    Race:
        Unique race identifier used for grouped validation.
    """

    df = df.copy()

    best = df.groupby(["Year", "Event", "Driver"])["LapTimeSeconds"].transform("min")

    df["Delta"] = df["LapTimeSeconds"] - best

    total_laps = df.groupby(["Year", "Event"])["LapNumber"].transform("max")

    df["FuelPct"] = 1.0 - df["LapNumber"] / total_laps

    df["Race"] = df["Year"].astype(str) + "_" + df["Event"].astype(str)

    return df


# =========================================================
# RIDGE DESIGN
# =========================================================


def ridge_design(df, compounds):
    """
    Build the explicit Ridge design matrix.

    Model structure:

        Delta =
            compound intercept
            + compound-specific age effect
            + compound-specific age^2 effect
            + shared FuelPct effect
            + shared TrackTemp effect

    age^2 is scaled by 100 to keep the feature numerically
    well behaved while retaining interpretability.
    """

    X = pd.DataFrame(index=df.index)

    for compound in compounds:
        is_compound = (df["Compound"] == compound).astype(float)

        X[f"int[{compound}]"] = is_compound

        X[f"age[{compound}]"] = is_compound * df["TyreLife"]

        X[f"age2[{compound}]"] = is_compound * df["TyreLife"] ** 2 / 100.0

    X["FuelPct"] = df["FuelPct"]
    X["TrackTemp"] = df["TrackTemp"]

    return X


def make_ridge():
    """
    Ridge with no global intercept because the compound
    dummy columns provide the compound-specific intercepts.
    """

    return Ridge(
        alpha=1.0,
        fit_intercept=False,
    )


# =========================================================
# XGBOOST
# =========================================================


def make_xgb():
    """
    XGBoost regression model used for comparison.
    """

    return XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        n_jobs=4,
    )


# =========================================================
# GROUPED CROSS-VALIDATION
# =========================================================


def cv_scores(X, y, groups, make_model):
    """
    Evaluate a model using race-grouped cross-validation.

    Each race is held out as a complete test group.

    Returns:
        Per-race MAE, R² and median baseline metrics.
    """

    groups = pd.Series(
        groups,
        index=X.index,
    )

    unique_groups = groups.nunique()

    if unique_groups < 2:
        raise ValueError("At least two race groups are required for GroupKFold.")

    rows = []

    splitter = GroupKFold(n_splits=unique_groups)

    for train_idx, test_idx in splitter.split(
        X,
        y,
        groups,
    ):
        model = make_model()

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
        )

        predictions = model.predict(X.iloc[test_idx])

        baseline = np.full(
            len(test_idx),
            y.iloc[train_idx].median(),
        )

        rows.append(
            {
                "race": groups.iloc[test_idx].iloc[0],
                "mae": mean_absolute_error(
                    y.iloc[test_idx],
                    predictions,
                ),
                "r2": r2_score(
                    y.iloc[test_idx],
                    predictions,
                ),
                "base_mae": mean_absolute_error(
                    y.iloc[test_idx],
                    baseline,
                ),
                "base_r2": r2_score(
                    y.iloc[test_idx],
                    baseline,
                ),
            }
        )

    return pd.DataFrame(rows)


def print_folds(name, folds):
    """
    Print per-race validation results and their mean.
    """

    print(f"\n--- {name} (GroupKFold, one whole race held out per fold) ---")

    print(
        f"{'held-out race':<28}"
        f"{'MAE':>8}"
        f"{'R2':>8}"
        f"{'base MAE':>10}"
        f"{'base R2':>9}"
        f"{'beats?':>8}"
    )

    for _, row in folds.iterrows():
        beats = "yes" if row["mae"] < row["base_mae"] else "NO"

        print(
            f"{row['race']:<28}"
            f"{row['mae']:>8.3f}"
            f"{row['r2']:>8.3f}"
            f"{row['base_mae']:>10.3f}"
            f"{row['base_r2']:>9.3f}"
            f"{beats:>8}"
        )

    mean_values = folds.mean(numeric_only=True)

    print(
        f"{'MEAN':<28}"
        f"{mean_values['mae']:>8.3f}"
        f"{mean_values['r2']:>8.3f}"
        f"{mean_values['base_mae']:>10.3f}"
        f"{mean_values['base_r2']:>9.3f}"
    )

    print(
        f"{'std across folds':<28}{folds['mae'].std():>8.3f}{folds['r2'].std():>8.3f}"
    )

    return mean_values


# =========================================================
# DEGRADATION PLOT
# =========================================================


def make_plot(df, ridge, compounds):
    """
    Create before/after degradation plots.

    Left:
        Raw mean pace loss by tyre age.

    Right:
        Ridge fitted curves with FuelPct and TrackTemp
        fixed at their dataset means.
    """

    (
        fig,
        (
            ax_raw,
            ax_fit,
        ),
    ) = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
        sharey=True,
    )

    # -----------------------------------------------------
    # BEFORE: raw means
    # -----------------------------------------------------

    grouped = (
        df.groupby(["Compound", "TyreLife"])["Delta"]
        .agg(["mean", "size"])
        .reset_index()
    )

    grouped = grouped[grouped["size"] >= MIN_LAPS]

    for compound, group in grouped.groupby("Compound"):
        group = group.sort_values("TyreLife")

        ax_raw.plot(
            group["TyreLife"],
            group["mean"],
            marker="o",
            ms=3,
            label=compound,
        )

    ax_raw.set_title("BEFORE: Raw pace loss")

    ax_raw.set_xlabel("Tyre age (laps)")

    ax_raw.set_ylabel("Pace loss vs driver's best lap (s)")

    # -----------------------------------------------------
    # AFTER: Ridge fitted curves
    # -----------------------------------------------------

    mean_row = {
        "FuelPct": df["FuelPct"].mean(),
        "TrackTemp": df["TrackTemp"].mean(),
    }

    for compound in compounds:
        compound_ages = df.loc[
            df["Compound"] == compound,
            "TyreLife",
        ]

        if compound_ages.empty:
            continue

        max_age = int(compound_ages.quantile(0.98))

        if max_age < 1:
            continue

        ages = np.arange(
            1,
            max_age + 1,
        )

        grid = pd.DataFrame(
            {
                "TyreLife": ages,
                "Compound": compound,
                **mean_row,
            }
        )

        predictions = ridge.predict(
            ridge_design(
                grid,
                compounds,
            )
        )

        ax_fit.plot(
            ages,
            predictions,
            lw=2,
            label=compound,
        )

    ax_fit.set_title("AFTER: Ridge fit, controls held at mean")

    ax_fit.set_xlabel("Tyre age (laps)")

    # -----------------------------------------------------
    # Common styling
    # -----------------------------------------------------

    for axis in (
        ax_raw,
        ax_fit,
    ):
        axis.legend()
        axis.grid(alpha=0.2)

    fig.suptitle("Tyre degradation before and after fuel correction")

    fig.tight_layout()

    fig.savefig(
        OUT_PNG,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nsaved {OUT_PNG}")


# =========================================================
# EXPORT CURVE DATA
# =========================================================


def export_chart_csvs(
    df,
    ridge,
    compounds,
):
    """
    Export raw and fitted degradation curves for
    dashboard/web visualisation.
    """

    # -----------------------------------------------------
    # BEFORE: raw mean values
    # -----------------------------------------------------

    grouped = (
        df.groupby(["Compound", "TyreLife"])["Delta"]
        .agg(["mean", "size"])
        .reset_index()
    )

    grouped = grouped[grouped["size"] >= MIN_LAPS]

    for compound in compounds:
        data = grouped[grouped["Compound"] == compound][["TyreLife", "mean"]].copy()

        data = data.rename(columns={"mean": "PaceLoss"})

        filename = DATA_DIR / f"before_{compound.lower()}.csv"

        data.to_csv(
            filename,
            index=False,
        )

        print(f"saved {filename}")

    # -----------------------------------------------------
    # AFTER: Ridge predictions
    # -----------------------------------------------------

    mean_row = {
        "FuelPct": df["FuelPct"].mean(),
        "TrackTemp": df["TrackTemp"].mean(),
    }

    for compound in compounds:
        compound_ages = df.loc[
            df["Compound"] == compound,
            "TyreLife",
        ]

        if compound_ages.empty:
            continue

        max_age = int(compound_ages.quantile(0.98))

        if max_age < 1:
            continue

        ages = np.arange(
            1,
            max_age + 1,
        )

        grid = pd.DataFrame(
            {
                "TyreLife": ages,
                "Compound": compound,
                **mean_row,
            }
        )

        predictions = ridge.predict(
            ridge_design(
                grid,
                compounds,
            )
        )

        data = pd.DataFrame(
            {
                "TyreLife": ages,
                "PaceLoss": predictions,
            }
        )

        filename = DATA_DIR / f"after_{compound.lower()}.csv"

        data.to_csv(
            filename,
            index=False,
        )

        print(f"saved {filename}")


# =========================================================
# MAIN
# =========================================================


def main():

    # =====================================================
    # 1. LOAD DATA
    # =====================================================

    if not IN_CSV.exists():
        raise FileNotFoundError(f"{IN_CSV} not found. Run fetch_data.py first.")

    df = add_features(pd.read_csv(IN_CSV))

    if df.empty:
        raise RuntimeError(f"{IN_CSV} loaded empty.")

    compounds = sorted(df["Compound"].dropna().unique().tolist())

    print(
        f"{len(df)} laps, "
        f"{df['Race'].nunique()} races, "
        f"compounds: {', '.join(compounds)}"
    )

    print("laps per compound:\n" + df["Compound"].value_counts().to_string())

    # =====================================================
    # 2. MODEL MATRICES
    # =====================================================

    y = df["Delta"]

    X_ridge = ridge_design(
        df,
        compounds,
    )

    xgb_input = df[XGB_FEATURES + ["Compound"]]

    X_xgb = pd.get_dummies(
        xgb_input,
        columns=["Compound"],
        dtype=float,
    )

    # Make sure all required values are valid.
    for name, X in (
        ("ridge", X_ridge),
        ("xgb", X_xgb),
    ):
        bad_columns = X.columns[X.isna().any()].tolist()

        assert not bad_columns, f"NaNs leaked into {name} features: {bad_columns}"

    assert not y.isna().any(), "NaNs in target Delta"

    # =====================================================
    # 3. IDENTIFY WET RACES
    # =====================================================

    per_race = df.groupby("Race").agg(
        rain=("Rainfall", "mean"),
        spread=("Delta", "std"),
        laps=("Delta", "size"),
    )

    wet = per_race.index[per_race["rain"] > WET_FRACTION].tolist()

    print(f"\n--- per race (wet = rain on >{WET_FRACTION:.0%} of clean laps) ---")

    print(f"{'race':<28}{'laps':>7}{'rain frac':>11}{'Delta std':>11}{'':>6}")

    for race, row in per_race.iterrows():
        wet_label = "  WET" if race in wet else ""

        print(
            f"{race:<28}"
            f"{int(row['laps']):>7}"
            f"{row['rain']:>11.3f}"
            f"{row['spread']:>11.3f}"
            f"{wet_label:>6}"
        )

    # =====================================================
    # 4. GROUPED CROSS-VALIDATION
    # =====================================================

    views = {
        "pooled": pd.Series(
            True,
            index=df.index,
        ),
        "dry only": ~df["Race"].isin(wet),
    }

    means = {}

    for view, mask in views.items():
        if df.loc[mask, "Race"].nunique() < 2:
            print(f"\nSkipping {view}: fewer than two race groups.")
            continue

        for name, X, make_model in (
            (
                "Ridge",
                X_ridge,
                make_ridge,
            ),
            (
                "XGBoost",
                X_xgb,
                make_xgb,
            ),
        ):
            folds = cv_scores(
                X[mask.values],
                y[mask.values],
                df.loc[
                    mask.values,
                    "Race",
                ],
                make_model,
            )

            means[(name, view)] = print_folds(
                f"{name}, {view}",
                folds,
            )

    # =====================================================
    # 5. FULL-DATA RIDGE REFIT
    # =====================================================

    # CV measures predictive performance.
    # This full-data refit is used for interpretable
    # coefficients and fitted degradation curves.

    ridge = make_ridge().fit(
        X_ridge,
        y,
    )

    coef = pd.Series(
        ridge.coef_,
        index=X_ridge.columns,
    )

    # =====================================================
    # 6. COMPOUND DEGRADATION GRADIENT
    # =====================================================

    net = {}

    for compound in compounds:
        compound_ages = df.loc[
            df["Compound"] == compound,
            "TyreLife",
        ]

        if compound_ages.empty:
            net[compound] = np.nan
            continue

        hi = compound_ages.quantile(0.98)

        b1 = coef[f"age[{compound}]"]

        b2 = coef[f"age2[{compound}]"]

        if hi <= 1:
            net[compound] = np.nan
            continue

        net[compound] = ((b1 * hi + b2 * hi**2 / 100.0) - (b1 + b2 / 100.0)) / (hi - 1)

    print("\n--- Ridge coefficients (full-data refit, quotable) ---")

    print(
        f"{'compound':<16}"
        f"{'intercept':>11}"
        f"{'age s/lap':>12}"
        f"{'age^2/100':>12}"
        f"{'net s/lap':>12}"
    )

    for compound in compounds:
        print(
            f"{compound:<16}"
            f"{coef[f'int[{compound}]']:>11.3f}"
            f"{coef[f'age[{compound}]']:>12.4f}"
            f"{coef[f'age2[{compound}]']:>12.4f}"
            f"{net[compound]:>12.4f}"
        )

    print(
        "net s/lap = average gradient of the fitted "
        "curve over that compound's observed age range"
    )

    print(
        f"\nfuel effect  {coef['FuelPct']:+.3f} s across a full tank (FuelPct 1 -> 0)"
    )

    print(f"track temp   {coef['TrackTemp']:+.4f} s per degC")

    # =====================================================
    # 7. EXPORT VISUALISATION DATA
    # =====================================================

    make_plot(
        df,
        ridge,
        compounds,
    )

    export_chart_csvs(
        df,
        ridge,
        compounds,
    )

    # =====================================================
    # 8. SUMMARY
    # =====================================================

    print("\n" + "=" * 62)

    checks = [
        (
            f"rows loaded ({len(df)})",
            len(df) > 0,
        ),
        (
            "no NaNs in features/target",
            True,
        ),
    ]

    for (name, view), mean_values in means.items():
        checks.append(
            (
                f"{name} {view}: "
                f"MAE {mean_values['mae']:.3f} "
                f"< baseline "
                f"{mean_values['base_mae']:.3f}",
                mean_values["mae"] < mean_values["base_mae"],
            )
        )

    for label, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")

    print("=" * 62)

    passed = sum(ok for _, ok in checks)

    print(f"{passed}/{len(checks)} checks passed")

    print(
        f"caveat: {df['Race'].nunique()} races "
        "form the CV groups, so per-fold metrics "
        "are noisy by construction."
    )

    print(
        "note: fitted degradation gradients are "
        "reported as diagnostics, not as a hard "
        "validation requirement."
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
