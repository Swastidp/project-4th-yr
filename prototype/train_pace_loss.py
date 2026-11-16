from pathlib import Path

import warnings

import numpy as np

import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.linear_model import HuberRegressor

from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.model_selection import GroupKFold

from sklearn.pipeline import Pipeline

# =========================================================

# SETTINGS

# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "training_data_tif1"

TRAINING_YEARS = [2022, 2023, 2024, 2025]

DRY_COMPOUNDS = ["HARD", "MEDIUM", "SOFT"]

BASELINE_START_INDEX = 2  # Zero-based index: stint lap 3*

BASELINE_LAPS = 7  # Stint laps 3–9 inclusive*

PREDICTION_START_INDEX = 9  # Zero-based index: predictions start at lap 10*

MIN_STINT_LAPS = 10

HUBER_EPSILON = 1.35
HUBER_ALPHA = 1.0
HUBER_MAX_ITER = 500

CV_SPLITS = 5

MAX_SUPPORTED_AGE_PERCENTILE = 98

REQUIRED_COLUMNS = [
    "Year",
    "Event",
    "Driver",
    "Stint",
    "LapNumber",
    "LapTimeSeconds",
    "TyreLife",
    "Compound",
    "Rainfall",
]

# =========================================================

# WEATHER CLASSIFICATION

# =========================================================


def classify_rainfall(value):
    """

    True  = wet

    False = dry

    None  = unknown

    """

    if pd.isna(value):
        return None

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    value = str(value).strip().lower()

    if value in ["true", "1", "yes"]:
        return True

    if value in ["false", "0", "no"]:
        return False

    return None


def classify_stint_weather(values):
    """

    A stint is dry only if every recorded lap is explicitly dry.

    """

    statuses = [classify_rainfall(value) for value in values]

    if True in statuses:
        return "wet"

    if None in statuses:
        return "unknown"

    return "dry"


# =========================================================

# LOAD PARQUET DATASET

# =========================================================


def load_training_data():

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Training data directory not found: {DATA_DIR}")

    files = sorted(DATA_DIR.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No Parquet files found in {DATA_DIR}")

    print("\n" + "=" * 75)

    print("LOADING HISTORICAL F1 TRAINING DATA")

    print("=" * 75)

    print(f"Dataset directory: {DATA_DIR}")

    print(f"Parquet files found: {len(files)}")

    frames = []

    failed_files = []

    for index, file in enumerate(files, start=1):
        try:
            frame = pd.read_parquet(file, engine="pyarrow")

            if frame.empty:
                failed_files.append((file.name, "Empty Parquet file"))

                continue

            frames.append(frame)

        except Exception as error:
            failed_files.append((file.name, str(error)))

        if index % 15 == 0 or index == len(files):
            print(
                f"Loaded {index}/{len(files)} files",
                flush=True,
            )

    if not frames:
        raise RuntimeError("No usable Parquet files could be loaded.")

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]

    if missing:
        raise ValueError(f"Training dataset missing columns: {missing}")

    # Verify unique race identity.

    data["Year"] = pd.to_numeric(
        data["Year"],
        errors="coerce",
    )

    data = data.dropna(subset=["Year", "Event"]).copy()

    data["Year"] = data["Year"].astype(int)

    data = data[data["Year"].isin(TRAINING_YEARS)].copy()

    # Build a unique race identifier.

    data["Race"] = data["Year"].astype(str) + "_" + data["Event"].astype(str)

    race_count = data["Race"].nunique()

    print("\nDATASET SUMMARY")

    print("-" * 75)

    print(f"Successfully loaded files: {len(frames)}")

    print(f"Failed files:              {len(failed_files)}")

    print(f"Unique races:              {race_count}")

    print(f"Total lap records:         {len(data):,}")

    print("\nSEASON BREAKDOWN")

    for year in TRAINING_YEARS:
        season = data[data["Year"] == year]

        print(f"{year}: {season['Event'].nunique()} races | {len(season):,} laps")

    print("\nCOMPOUND DISTRIBUTION")

    print(data["Compound"].value_counts().to_string())

    if failed_files:
        print("\nFILES THAT COULD NOT BE LOADED")

        for filename, reason in failed_files:
            print(f"{filename}: {reason}")

    if race_count < CV_SPLITS:
        raise ValueError(
            f"Only {race_count} races available. At least {CV_SPLITS} are required."
        )

    return data


# =========================================================

# DATA PREPARATION

# =========================================================


def prepare_data(data):

    df = data.copy()

    # -----------------------------------------------------

    # NUMERIC CONVERSION

    # -----------------------------------------------------

    numeric_columns = [
        "Year",
        "Stint",
        "LapNumber",
        "LapTimeSeconds",
        "TyreLife",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["Compound"] = df["Compound"].astype("string").str.upper().str.strip()

    # -----------------------------------------------------

    # VALIDATION

    # -----------------------------------------------------

    df = df.dropna(
        subset=[
            "Year",
            "Event",
            "Driver",
            "Stint",
            "LapNumber",
            "LapTimeSeconds",
            "TyreLife",
            "Compound",
        ]
    ).copy()

    df = df[
        (df["LapTimeSeconds"] > 0)
        & (df["LapNumber"] > 0)
        & (df["TyreLife"] > 0)
        & (df["Stint"] > 0)
    ].copy()

    # -----------------------------------------------------

    # COMPOUND FILTER

    # -----------------------------------------------------

    before_compound = len(df)

    df = df[df["Compound"].isin(DRY_COMPOUNDS)].copy()

    excluded_compound = before_compound - len(df)

    if df.empty:
        raise ValueError("No HARD, MEDIUM or SOFT laps available.")

    # -----------------------------------------------------

    # IDENTIFY ORIGINAL STINTS

    # -----------------------------------------------------

    df["Race"] = df["Year"].astype(int).astype(str) + "_" + df["Event"].astype(str)

    df["StintGroup"] = (
        df["Race"]
        + "_"
        + df["Driver"].astype(str)
        + "_"
        + df["Stint"].astype(int).astype(str)
    )

    # -----------------------------------------------------

    # WHOLE-STINT WEATHER FILTER

    # -----------------------------------------------------

    weather_status = df.groupby("StintGroup")["Rainfall"].apply(classify_stint_weather)

    wet_stints = set(weather_status[weather_status == "wet"].index)

    unknown_stints = set(weather_status[weather_status == "unknown"].index)

    dry_stints = set(weather_status[weather_status == "dry"].index)

    before_weather = len(df)

    df = df[df["StintGroup"].isin(dry_stints)].copy()

    excluded_weather_laps = before_weather - len(df)

    if df.empty:
        raise ValueError("No fully dry stints remain.")

    # -----------------------------------------------------

    # CHECK STINT COMPOUND CONSISTENCY

    # -----------------------------------------------------

    mixed = df.groupby("StintGroup")["Compound"].nunique()

    if (mixed > 1).any():
        raise ValueError("Some original stints contain multiple compounds.")

    # -----------------------------------------------------

    # SORT

    # -----------------------------------------------------

    df = df.sort_values(
        [
            "Year",
            "Event",
            "Driver",
            "Stint",
            "LapNumber",
        ]
    ).reset_index(drop=True)

    # -----------------------------------------------------

    # CLEANING SUMMARY

    # -----------------------------------------------------

    print("\nDATA CLEANING SUMMARY")

    print("-" * 75)

    actual_years = sorted(df["Year"].dropna().astype(int).unique().tolist())
    print(f"Training seasons:               {actual_years}")

    print(f"Non-dry compound laps excluded: {excluded_compound:,}")

    print(f"Wet stints excluded:            {len(wet_stints):,}")

    print(f"Unknown-weather stints excluded:{len(unknown_stints):,}")

    print(f"Laps removed by weather filter: {excluded_weather_laps:,}")

    print(f"Fully dry stints retained:      {len(dry_stints):,}")

    print(f"Remaining dry training laps:    {len(df):,}")

    print(f"Remaining races:                {df['Race'].nunique()}")

    print("\nSEASON BREAKDOWN")

    for year in actual_years:
        season = df[df["Year"] == year]

        print(f"{year}: {season['Event'].nunique()} races | {len(season):,} laps")

    print("\nCOMPOUNDS")

    print(df["Compound"].value_counts().to_string())

    return df


# =========================================================

# BUILD STINT TARGETS

# =========================================================


def build_stint_targets(df):

    rows = []

    excluded_short = 0

    excluded_gaps = 0

    excluded_invalid_baselines = 0

    # Maximum race lap for each event.

    race_lengths = df.groupby("Race")["LapNumber"].max().to_dict()

    group_columns = [
        "Year",
        "Event",
        "Driver",
        "Stint",
        "StintGroup",
        "Race",
        "Compound",
    ]

    for _, stint in df.groupby(group_columns, sort=False):
        stint = stint.sort_values("LapNumber").copy()

        if len(stint) < MIN_STINT_LAPS:
            excluded_short += 1

            continue

        # The baseline requires the first nine stint laps to be consecutive.

        first_nine_laps = stint["LapNumber"].iloc[:PREDICTION_START_INDEX].to_numpy()

        if len(first_nine_laps) < PREDICTION_START_INDEX or not np.all(
            np.diff(first_nine_laps) == 1
        ):
            excluded_gaps += 1

            continue

        # Use the median of stint laps 3–9 as the seven-lap baseline.

        baseline_rows = stint.iloc[
            BASELINE_START_INDEX : BASELINE_START_INDEX + BASELINE_LAPS
        ]

        baseline_time = float(baseline_rows["LapTimeSeconds"].median())

        if not np.isfinite(baseline_time) or baseline_time <= 0:
            excluded_invalid_baselines += 1

            continue

        starting_age = float(stint["TyreLife"].iloc[0])

        race_length = race_lengths[stint["Race"].iloc[0]]

        for _, lap in stint.iloc[PREDICTION_START_INDEX:].iterrows():
            age = float(lap["TyreLife"]) - starting_age

            if age <= 0:
                continue

            lap_number = float(lap["LapNumber"])

            rows.append(
                {
                    "Race": lap["Race"],
                    "Year": int(lap["Year"]),
                    "Event": lap["Event"],
                    "Driver": lap["Driver"],
                    "Compound": lap["Compound"],
                    "StintGroup": lap["StintGroup"],
                    "StintAge": age,
                    "LapNumber": lap_number,
                    "RaceProgress": lap_number / race_length,
                    "Delta": float(lap["LapTimeSeconds"]) - baseline_time,
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError("No usable stint targets were generated.")

    print("\nSTINT PREPARATION")

    print("-" * 75)

    print("Baseline: median of stint laps 3–9 (7 laps)")

    print("Prediction starts: stint lap 10 onward")

    print(f"Stints found:                  {df['StintGroup'].nunique():,}")

    print(f"Short stints excluded:         {excluded_short:,}")

    print(f"Stints with baseline gaps:     {excluded_gaps:,}")

    print(f"Invalid baselines excluded:    {excluded_invalid_baselines:,}")

    print(f"Training samples:              {len(result):,}")

    return result


# =========================================================

# FEATURE ENGINEERING

# =========================================================


def build_features(df, compounds):

    X = pd.DataFrame(index=df.index)

    age = df["StintAge"]

    for compound in compounds:
        mask = (df["Compound"] == compound).astype(float)

        X[f"age_{compound}"] = mask * age

        X[f"age2_{compound}"] = mask * age.pow(2) / 100.0

    # Control for race progress, including nonlinear effects.

    progress = df["RaceProgress"].clip(0, 1)

    X["RaceProgress"] = progress

    X["RaceProgress2"] = progress.pow(2)

    return X.astype(float)


# =========================================================

# MODEL

# =========================================================


def make_model():

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "huber",
                HuberRegressor(
                    epsilon=HUBER_EPSILON,
                    alpha=HUBER_ALPHA,
                    max_iter=HUBER_MAX_ITER,
                    fit_intercept=False,
                ),
            ),
        ]
    )


# =========================================================

# RACE-GROUPED VALIDATION

# =========================================================


def validate_model(df, X, y):

    races = df["Race"]

    n_splits = min(
        CV_SPLITS,
        races.nunique(),
    )

    if n_splits < 2:
        raise ValueError("At least two races are required.")

    splitter = GroupKFold(
        n_splits=n_splits,
    )

    predictions = np.full(
        len(df),
        np.nan,
    )

    folds = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, races),
        start=1,
    ):
        model = make_model()

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
        )

        pred = model.predict(X.iloc[test_idx])

        predictions[test_idx] = pred

        folds.append(
            {
                "Fold": fold,
                "Races": races.iloc[test_idx].nunique(),
                "Laps": len(test_idx),
                "MAE": mean_absolute_error(
                    y.iloc[test_idx],
                    pred,
                ),
                "R2": r2_score(
                    y.iloc[test_idx],
                    pred,
                ),
            }
        )

    if np.isnan(predictions).any():
        raise RuntimeError("Some validation samples did not receive predictions.")

    oof = df.copy()

    oof["Actual"] = y.to_numpy()

    oof["Prediction"] = predictions

    oof["Error"] = (oof["Actual"] - oof["Prediction"]).abs()

    return oof, pd.DataFrame(folds)


# =========================================================

# VALIDATION RESULTS

# =========================================================


def print_results(oof, folds):

    mae = mean_absolute_error(
        oof["Actual"],
        oof["Prediction"],
    )

    zero_mae = mean_absolute_error(
        oof["Actual"],
        np.zeros(len(oof)),
    )

    r2 = r2_score(
        oof["Actual"],
        oof["Prediction"],
    )

    bias = (oof["Prediction"] - oof["Actual"]).mean()

    improvement = (zero_mae - mae) / zero_mae * 100 if zero_mae > 0 else np.nan

    print("\n" + "=" * 75)

    print("RACE-GROUPED MODEL VALIDATION")

    print("=" * 75)

    print(f"Validation samples:    {len(oof):,}")

    print(f"Validation races:      {oof['Race'].nunique()}")

    print(f"Huber MAE:             {mae:.3f} seconds")

    print(f"Zero baseline MAE:     {zero_mae:.3f} seconds")

    print(f"Improvement:           {improvement:+.2f}%")

    print(f"R²:                    {r2:.3f}")

    print(f"Prediction bias:       {bias:+.3f} seconds")

    print("\nFOLD RESULTS")

    print("-" * 75)

    print(folds.to_string(index=False))

    print("\nCOMPOUND RESULTS")

    print("-" * 75)

    for compound, group in oof.groupby("Compound"):
        compound_mae = mean_absolute_error(
            group["Actual"],
            group["Prediction"],
        )

        print(f"{compound:<12}Laps: {len(group):>7,}  MAE: {compound_mae:.3f}s")

    print("\nSEASON RESULTS")

    print("-" * 75)

    for year, group in oof.groupby("Year"):
        season_mae = mean_absolute_error(
            group["Actual"],
            group["Prediction"],
        )

        print(f"{int(year)}: {len(group):,} samples | MAE: {season_mae:.3f}s")

    if mae < zero_mae:
        print("\n[PASS] Model beats the zero baseline overall.")

    else:
        print("\n[FAIL] Model does not beat the zero baseline.")

    return {
        "mae": mae,
        "zero_baseline_mae": zero_mae,
        "improvement": improvement,
        "r2": r2,
        "bias": bias,
    }


# =========================================================

# COEFFICIENT DIAGNOSTICS

# =========================================================


def print_coefficient_diagnostics(
    model,
    compounds,
    supported_ages,
):

    huber = model.named_steps["huber"]

    print("\n" + "=" * 75)

    print("LEARNED HUBER DEGRADATION COEFFICIENTS")

    print("=" * 75)

    for name, value in zip(
        model.feature_names_in_,
        huber.coef_,
    ):
        print(f"{name:<25} {value:+.6f}")

    print("\nMODEL-PREDICTED TYRE AGE CURVES")

    print("Race progress held at 50%.")

    print("Predictions are seconds relative to the stint baseline.")

    print("-" * 75)

    for compound in compounds:
        max_age = supported_ages[compound]

        print(f"\n{compound} (supported age up to {max_age:.1f} laps)")

        ages = sorted(
            set(
                [
                    0,
                    5,
                    10,
                    15,
                    20,
                    25,
                    int(max_age),
                ]
            )
        )

        for age in ages:
            if age > max_age:
                continue

            row = {
                "age_HARD": 0.0,
                "age2_HARD": 0.0,
                "age_MEDIUM": 0.0,
                "age2_MEDIUM": 0.0,
                "age_SOFT": 0.0,
                "age2_SOFT": 0.0,
                "RaceProgress": 0.5,
                "RaceProgress2": 0.25,
            }

            row[f"age_{compound}"] = float(age)

            row[f"age2_{compound}"] = float(age**2) / 100.0

            feature_row = pd.DataFrame(
                [row],
                columns=model.feature_names_in_,
            )

            prediction = model.predict(feature_row)[0]

            print(f"Tyre age {age:>2}: {prediction:+.3f}s")


# =========================================================

# OBSERVED VS PREDICTED BY TYRE AGE

# =========================================================


def print_observed_age_comparison(oof):

    print("\n" + "=" * 75)

    print("OBSERVED VS PREDICTED DEGRADATION BY TYRE AGE")

    print("=" * 75)

    bins = [0, 5, 10, 15, 20, 25, 30, 100]

    labels = [
        "0-4",
        "5-9",
        "10-14",
        "15-19",
        "20-24",
        "25-29",
        "30+",
    ]

    data = oof.copy()

    data["AgeBand"] = pd.cut(
        data["StintAge"],
        bins=bins,
        labels=labels,
        right=False,
    )

    for compound in DRY_COMPOUNDS:
        group = data[data["Compound"] == compound]

        if group.empty:
            continue

        print(f"\n{compound}")

        print(f"{'Age':<10}{'Laps':>8}{'Actual':>12}{'Predicted':>13}")

        for age_band, age_group in group.groupby(
            "AgeBand",
            observed=True,
        ):
            if age_group.empty:
                continue

            print(
                f"{str(age_band):<10}"
                f"{len(age_group):>8,}"
                f"{age_group['Actual'].mean():>11.3f}s"
                f"{age_group['Prediction'].mean():>12.3f}s"
            )


# =========================================================

# TRAINING PIPELINE

# =========================================================


def train_pace_loss(training_data):

    cleaned = prepare_data(training_data)
    actual_training_years = sorted(cleaned["Year"].dropna().astype(int).unique().tolist())

    stints = build_stint_targets(cleaned)

    compounds = [
        compound for compound in DRY_COMPOUNDS if compound in set(stints["Compound"])
    ]

    if not compounds:
        raise ValueError("No dry compounds available for training.")

    # -----------------------------------------------------

    # FEATURE ENGINEERING

    # -----------------------------------------------------

    X = build_features(
        stints,
        compounds,
    )

    y = stints["Delta"].astype(float)

    # -----------------------------------------------------

    # SUPPORTED TYRE AGE

    # -----------------------------------------------------

    supported_ages = {}

    for compound in compounds:
        ages = stints.loc[
            stints["Compound"] == compound,
            "StintAge",
        ]

        supported_ages[compound] = float(
            ages.quantile(MAX_SUPPORTED_AGE_PERCENTILE / 100)
        )

    # -----------------------------------------------------

    # CROSS-VALIDATION

    # -----------------------------------------------------

    oof, folds = validate_model(
        stints,
        X,
        y,
    )

    metrics = print_results(
        oof,
        folds,
    )

    # -----------------------------------------------------

    # FINAL MODEL

    # -----------------------------------------------------

    final_model = make_model()

    final_model.fit(
        X,
        y,
    )

    print_coefficient_diagnostics(
        final_model,
        compounds,
        supported_ages,
    )

    print_observed_age_comparison(oof)

    print("\n" + "=" * 75)

    print("FINAL HUBER PACE-LOSS MODEL")

    print("=" * 75)

    print(f"Training seasons: {actual_training_years}")

    print(f"Training races:   {stints['Race'].nunique()}")

    print(f"Training samples: {len(stints):,}")

    print(f"Compounds:        {', '.join(compounds)}")

    print("\nSUPPORTED TYRE AGES")

    for compound, age in supported_ages.items():
        print(f"{compound}: {age:.1f} laps (98th percentile)")

    print("\nModel fitted on all eligible training samples.")

    print("No CSV or plot files created.")

    return {
        "model": final_model,
        "compounds": compounds,
        "feature_columns": X.columns.tolist(),
        "metrics": metrics,
        "fold_metrics": folds,
        "oof_predictions": oof,
        "training_data": stints,
        "supported_ages": supported_ages,
        "training_years": actual_training_years,
    }


# =========================================================

# MAIN

# =========================================================


def main():

    data = load_training_data()

    return train_pace_loss(data)


# =========================================================

# RUN

# =========================================================

if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    TRAINED_PACE_MODEL = main()
