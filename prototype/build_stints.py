from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from lifelines import KaplanMeierFitter, WeibullAFTFitter
from lifelines.utils import concordance_index

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "training_data_tif1"

YEARS = [2022, 2023, 2024, 2025]
DRY_COMPOUNDS = ["HARD", "MEDIUM", "SOFT"]

TEST_SIZE = 0.20
RANDOM_STATE = 42

WEATHER_COLUMNS = [
    "TrackTemp",
    "AirTemp",
    "Humidity",
    "Pressure",
    "WindSpeed",
]

REQUIRED_COLUMNS = [
    "Year",
    "Event",
    "Driver",
    "Stint",
    "LapNumber",
    "Compound",
    "TyreLife",
    "Rainfall",
]


# =========================================================
# DATA LOADING
# =========================================================


def load_training_data():
    files = sorted(DATA_DIR.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No Parquet files found in: {DATA_DIR}")

    print("\n" + "=" * 75)
    print("LOADING LOCAL F1 DATA")
    print("=" * 75)
    print(f"Parquet files: {len(files)}")

    frames = []

    for index, file in enumerate(files, start=1):
        frame = pd.read_parquet(file)

        if not frame.empty:
            frames.append(frame)

        if index % 15 == 0 or index == len(files):
            print(f"Loaded {index}/{len(files)} files")

    if not frames:
        raise RuntimeError("No usable Parquet data found.")

    data = pd.concat(frames, ignore_index=True)

    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["Year", "LapNumber", "Stint", "TyreLife"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["Compound"] = data["Compound"].astype("string").str.upper().str.strip()

    data = data.dropna(subset=["Year", "Event", "Driver", "Stint", "LapNumber"]).copy()

    data["Year"] = data["Year"].astype(int)
    data = data[data["Year"].isin(YEARS)].copy()

    data["Race"] = data["Year"].astype(str) + "_" + data["Event"].astype(str)

    print(f"\nTotal lap records: {len(data):,}")
    print(f"Unique races: {data['Race'].nunique()}")

    print("\nSEASON BREAKDOWN")
    for year in YEARS:
        season = data[data["Year"] == year]
        print(f"{year}: {season['Event'].nunique()} races | {len(season):,} laps")

    return data


# =========================================================
# RAINFALL CLASSIFICATION
# =========================================================


def classify_rainfall(values):
    statuses = []

    for value in values:
        if pd.isna(value):
            statuses.append(None)
            continue

        if isinstance(value, (bool, np.bool_)):
            statuses.append(bool(value))
            continue

        value = str(value).strip().lower()

        if value in ("true", "1", "yes"):
            statuses.append(True)
        elif value in ("false", "0", "no"):
            statuses.append(False)
        else:
            statuses.append(None)

    if True in statuses:
        return True

    if None in statuses:
        return None

    return False


# =========================================================
# LAP PREPARATION
# =========================================================


def prepare_laps(data):
    df = data.copy()

    for col in ["LapNumber", "Stint", "TyreLife"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Compound"] = df["Compound"].astype("string").str.upper().str.strip()

    df = df.dropna(
        subset=[
            "Year",
            "Event",
            "Driver",
            "Stint",
            "LapNumber",
            "Compound",
        ]
    ).copy()

    df = df[(df["LapNumber"] > 0) & (df["Stint"] > 0)].copy()

    df["Race"] = df["Year"].astype(str) + "_" + df["Event"].astype(str)

    df = df.sort_values(["Year", "Event", "Driver", "Stint", "LapNumber"]).reset_index(
        drop=True
    )

    return df


# =========================================================
# BUILD ONE ROW PER ORIGINAL STINT
# =========================================================


def build_stint_observations(laps):
    print("\nBUILDING ORIGINAL TYRE STINTS")
    print("-" * 75)

    race_lengths = laps.groupby("Race")["LapNumber"].max().to_dict()

    group_columns = [
        "Year",
        "Event",
        "Race",
        "Driver",
        "Stint",
    ]

    aggregations = {
        "start_lap": ("LapNumber", "min"),
        "end_lap": ("LapNumber", "max"),
        # Elapsed lap span includes any missing recorded lap numbers.
        "stint_length": (
            "LapNumber",
            lambda x: x.max() - x.min() + 1,
        ),
        # Count distinct recorded lap numbers separately from elapsed span.
        "recorded_lap_rows": ("LapNumber", "nunique"),
        "raw_lap_rows": ("LapNumber", "size"),
        "starting_tyre_age": ("TyreLife", "first"),
        "ending_tyre_age": ("TyreLife", "last"),
        "Compound": ("Compound", "first"),
        "n_compounds": ("Compound", "nunique"),
        "Rainfall": ("Rainfall", classify_rainfall),
    }

    for col in WEATHER_COLUMNS:
        if col in laps.columns:
            aggregations[col] = (col, "first")

    stints = laps.groupby(group_columns, dropna=True).agg(**aggregations).reset_index()

    if stints.empty:
        raise RuntimeError("No stint observations found.")

    if (stints["n_compounds"] > 1).any():
        raise ValueError("At least one stint contains multiple compounds.")

    stints["total_race_laps"] = stints["Race"].map(race_lengths)

    # Missing records are the difference between the elapsed lap span
    # and the number of distinct lap numbers actually present.
    stints["missing_lap_count"] = (
        (stints["stint_length"] - stints["recorded_lap_rows"]).clip(lower=0).astype(int)
    )

    # Duplicate rows are reported separately and do not inflate recorded laps.
    stints["duplicate_lap_rows"] = (
        (stints["raw_lap_rows"] - stints["recorded_lap_rows"]).clip(lower=0).astype(int)
    )

    # This means the stint reached the final observed race-distance window;
    # it does not prove the driver was officially classified as a finisher.
    stints["finished_race"] = stints["end_lap"] >= stints["total_race_laps"] - 1

    print(f"Original stints: {len(stints):,}")
    print(
        "Stints with multiple compounds:",
        int((stints["n_compounds"] > 1).sum()),
    )
    print(
        "Stints with missing recorded laps:",
        int((stints["missing_lap_count"] > 0).sum()),
    )
    print(
        "Total missing recorded laps:",
        int(stints["missing_lap_count"].sum()),
    )
    print(
        "Duplicate lap rows:",
        int(stints["duplicate_lap_rows"].sum()),
    )

    return stints


# =========================================================
# EVENT / CENSORING LABEL
# =========================================================


def add_censoring(stints):
    stints = stints.copy()

    group_columns = ["Year", "Event", "Driver"]
    max_stint = stints.groupby(group_columns)["Stint"].transform("max")

    # 1 = another stint followed (observed transition event).
    # 0 = final observed stint for that driver in that race (right-censored).
    stints["event"] = (stints["Stint"] < max_stint).astype(int)

    # Preserve why a final stint is censored where the available data allows.
    # An early end is deliberately NOT labelled as a retirement: the lap data
    # alone cannot reliably distinguish retirement from incomplete records.
    stints["censoring_reason"] = np.select(
        [
            stints["event"].eq(1),
            stints["event"].eq(0) & stints["finished_race"].fillna(False),
            stints["event"].eq(0),
        ],
        [
            "another_stint_followed",
            "reached_final_observed_race_lap",
            "early_end_or_retirement_unknown",
        ],
        default="unknown",
    )

    stints["is_right_censored"] = stints["event"].eq(0)

    return stints


# =========================================================
# DRY STINT FILTER
# =========================================================


def filter_dry_stints(stints):
    stints = stints.copy()
    original_count = len(stints)

    stints["Compound"] = stints["Compound"].astype("string").str.upper().str.strip()

    compound_mask = stints["Compound"].isin(DRY_COMPOUNDS)
    excluded_compounds = int((~compound_mask).sum())
    stints = stints[compound_mask].copy()

    wet_mask = stints["Rainfall"].map(
        lambda x: (
            (isinstance(x, (bool, np.bool_)) and bool(x))
            or (isinstance(x, str) and x.strip().lower() in ("true", "1", "yes"))
        )
    )

    dry_mask = stints["Rainfall"].map(
        lambda x: (
            (isinstance(x, (bool, np.bool_)) and not bool(x))
            or (isinstance(x, str) and x.strip().lower() in ("false", "0", "no"))
        )
    )

    excluded_wet = int(wet_mask.sum())
    excluded_unknown = int((~(wet_mask | dry_mask)).sum())

    stints = stints[dry_mask].copy()

    numeric_columns = [
        "stint_length",
        "starting_tyre_age",
        "ending_tyre_age",
    ]

    for col in numeric_columns:
        stints[col] = pd.to_numeric(stints[col], errors="coerce")

    before_validation = len(stints)

    stints = stints.dropna(subset=numeric_columns).copy()

    stints = stints[
        (stints["stint_length"] > 0)
        & (stints["starting_tyre_age"] > 0)
        & (stints["ending_tyre_age"] > 0)
    ].copy()

    excluded_invalid = before_validation - len(stints)

    if stints.empty:
        raise RuntimeError("No valid dry stints remain.")

    print("\nDRY STINT FILTERING")
    print("-" * 75)
    print(f"Original stints:            {original_count:,}")
    print(f"Non-dry compounds excluded: {excluded_compounds:,}")
    print(f"Wet stints excluded:        {excluded_wet:,}")
    print(f"Unknown weather excluded:   {excluded_unknown:,}")
    print(f"Invalid stints excluded:    {excluded_invalid:,}")
    print(f"Remaining dry stints:       {len(stints):,}")

    return stints.reset_index(drop=True)


# =========================================================
# KAPLAN-MEIER DESCRIPTIVE BASELINE
# =========================================================


def fit_kaplan_meier(stints):
    print("\nKAPLAN-MEIER SURVIVAL ESTIMATES")
    print("-" * 75)

    rows = []
    fitters = {}

    for compound in DRY_COMPOUNDS:
        group = stints[stints["Compound"] == compound].copy()

        if group.empty:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=group["stint_length"],
            event_observed=group["event"],
            label=compound,
        )

        fitters[compound] = kmf
        median = kmf.median_survival_time_

        rows.append(
            {
                "Compound": compound,
                "Stints": len(group),
                "Events": int(group["event"].sum()),
                "Censored": int((group["event"] == 0).sum()),
                "Raw median": group["stint_length"].median(),
                "KM median": (float(median) if np.isfinite(median) else np.nan),
            }
        )

    summary = pd.DataFrame(rows)

    print(
        summary.to_string(
            index=False,
            formatters={
                "Raw median": "{:.1f}".format,
                "KM median": (lambda x: f"{x:.1f}" if pd.notna(x) else "Not reached"),
            },
        )
    )

    return summary, fitters


# =========================================================
# SHARED RACE-LEVEL TRAIN / TEST SPLIT
# =========================================================


def split_by_race(data, test_size=TEST_SIZE):
    unique_races = data["Race"].dropna().unique()

    if len(unique_races) < 5:
        raise RuntimeError("Too few races for a race-grouped holdout.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(splitter.split(data, groups=data["Race"]))

    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    print("\nRACE-GROUPED TRAIN / TEST SPLIT")
    print("-" * 75)
    print(f"Training races: {train['Race'].nunique()}")
    print(f"Testing races:  {test['Race'].nunique()}")
    print(f"Training rows:  {len(train):,}")
    print(f"Testing rows:   {len(test):,}")

    return train, test


# =========================================================
# MODEL A — WEIBULL AFT: STINT DURATION
# =========================================================


def prepare_aft_data(stints, fit_imputer=True, imputation_values=None):
    """Prepare Weibull AFT features.

    For evaluation, call with fit_imputer=False, split by race, then
    calculate medians from the training partition only and apply them to
    both partitions. The default keeps the full-data refit interface used
    by recommend.py, where fitting on all historical data is intentional.
    """
    data = stints.copy()

    data["RaceProgress"] = pd.to_numeric(
        data["start_lap"], errors="coerce"
    ) / pd.to_numeric(data["total_race_laps"], errors="coerce")
    data["RaceProgressSquared"] = data["RaceProgress"] ** 2

    numeric_features = [
        "starting_tyre_age",
        "RaceProgress",
        "RaceProgressSquared",
        *[col for col in WEATHER_COLUMNS if col in data.columns],
    ]

    for col in numeric_features:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["Year"] = data["Year"].astype(str)
    data["Compound"] = data["Compound"].astype(str).str.upper().str.strip()

    data["duration"] = pd.to_numeric(data["stint_length"], errors="coerce")
    data["event"] = pd.to_numeric(data["event"], errors="coerce")

    required = [
        "duration",
        "event",
        "Race",
        "Compound",
        "Year",
    ]

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required).copy()
    data = data[data["duration"] > 0].copy()

    if imputation_values is not None:
        for col in numeric_features:
            fill_value = imputation_values.get(col, 0.0)
            data[col] = data[col].fillna(fill_value)
    elif fit_imputer:
        # Used for the final full-history refit, not holdout evaluation.
        medians = data[numeric_features].median()
        medians = medians.fillna(0.0)
        for col in numeric_features:
            data[col] = data[col].fillna(float(medians[col]))

    return data, numeric_features


def fit_weibull_aft(stints):
    print("\n" + "=" * 75)
    print("MODEL A — WEIBULL AFT STINT DURATION")
    print("=" * 75)

    # Do not calculate imputation medians before the race-level split.
    data, numeric_features = prepare_aft_data(
        stints,
        fit_imputer=False,
    )
    train, test = split_by_race(data)

    # Fit imputation values on training races only, then reuse them for test.
    imputation_values = train[numeric_features].median().fillna(0.0)
    for col in numeric_features:
        fill_value = float(imputation_values[col])
        train[col] = train[col].fillna(fill_value)
        test[col] = test[col].fillna(fill_value)

    # Formula keeps compound and year categorical.
    formula_terms = [
        "C(Compound)",
        "C(Year)",
        *numeric_features,
    ]
    formula = " + ".join(formula_terms)

    aft = WeibullAFTFitter(penalizer=0.01)

    aft.fit(
        train,
        duration_col="duration",
        event_col="event",
        formula=formula,
    )

    print("\nWEIBULL AFT MODEL SUMMARY")
    print("-" * 75)

    summary_columns = [
        "coef",
        "exp(coef)",
        "se(coef)",
        "coef lower 95%",
        "coef upper 95%",
        "p",
    ]

    print(aft.summary[summary_columns].to_string(float_format=lambda x: f"{x:.4f}"))

    predicted_median = aft.predict_median(test)
    predicted_median = np.asarray(predicted_median).reshape(-1)

    test_durations = test["duration"].to_numpy(dtype=float)
    test_events = test["event"].to_numpy(dtype=int)

    valid = (
        np.isfinite(predicted_median)
        & np.isfinite(test_durations)
        & (predicted_median > 0)
    )

    aft_c_index = concordance_index(
        test_durations[valid],
        predicted_median[valid],
        test_events[valid],
    )

    observed_mask = valid & (test_events == 1)

    if observed_mask.any():
        observed_mae = np.mean(
            np.abs(test_durations[observed_mask] - predicted_median[observed_mask])
        )
    else:
        observed_mae = np.nan

    print("\nWEIBULL AFT HOLDOUT PERFORMANCE")
    print("-" * 75)
    print(f"Test stints: {len(test):,}")
    print(f"Test races: {test['Race'].nunique()}")
    print(f"Test concordance index: {aft_c_index:.4f}")
    print(
        "MAE on observed transitions only: "
        + (f"{observed_mae:.2f} laps" if np.isfinite(observed_mae) else "N/A")
    )

    print("\nEXAMPLE MEDIAN STINT DURATION PREDICTIONS")
    examples = test[["Race", "Driver", "Compound", "duration", "event"]].copy()
    examples["PredictedMedianLaps"] = predicted_median
    print(
        examples.head(10).to_string(
            index=False,
            formatters={
                "PredictedMedianLaps": "{:.1f}".format,
            },
        )
    )

    return aft, train, test, aft_c_index, observed_mae


# =========================================================
# MODEL B — LAP-LEVEL DISCRETE-TIME TRANSITION MODEL
# =========================================================


def build_lap_level_hazard_data(laps, eligible_stints):
    print("\nBUILDING LAP-LEVEL TRANSITION DATA")
    print("-" * 75)

    keys = [
        "Year",
        "Event",
        "Race",
        "Driver",
        "Stint",
    ]

    stint_labels = eligible_stints[keys + ["event"]].drop_duplicates(keys)

    lap_data = laps.merge(
        stint_labels,
        on=keys,
        how="inner",
        validate="many_to_one",
    )

    race_lengths = laps.groupby("Race")["LapNumber"].max().to_dict()
    lap_data["total_race_laps"] = lap_data["Race"].map(race_lengths)

    lap_data["RaceProgress"] = lap_data["LapNumber"] / lap_data["total_race_laps"]

    lap_data["TyreAge"] = pd.to_numeric(lap_data["TyreLife"], errors="coerce")

    lap_data = lap_data.dropna(
        subset=["LapNumber", "TyreAge", "Compound", "RaceProgress"]
    ).copy()

    lap_data = lap_data.sort_values(keys + ["LapNumber"]).reset_index(drop=True)

    # Each observed lap is a discrete risk interval.
    # A positive label is assigned only to the final recorded lap
    # of a stint when another stint follows.
    lap_data["transition_next_observed_lap"] = 0

    last_rows = lap_data.groupby(keys, sort=False).tail(1).index
    lap_data.loc[last_rows, "transition_next_observed_lap"] = lap_data.loc[
        last_rows, "event"
    ].astype(int)

    # Use current-lap features only.
    feature_columns = [
        "Compound",
        "Year",
        "TyreAge",
        "RaceProgress",
        *[col for col in WEATHER_COLUMNS if col in lap_data.columns],
    ]

    for col in WEATHER_COLUMNS:
        if col in lap_data.columns:
            lap_data[col] = pd.to_numeric(lap_data[col], errors="coerce")

    # Year is already included in `keys`; feature_columns also contains Year.
    # Deduplicate the selection to prevent duplicate DataFrame column names.
    result_columns = list(
        dict.fromkeys(
            keys
            + [
                "LapNumber",
                "transition_next_observed_lap",
                *feature_columns,
            ]
        )
    )
    result = lap_data[result_columns].copy()

    result["Year"] = result["Year"].astype(str)
    result["Compound"] = result["Compound"].astype(str).str.upper().str.strip()

    result = result.replace([np.inf, -np.inf], np.nan)

    print(f"Lap-level observations: {len(result):,}")
    print(
        "Recorded transition intervals:",
        int(result["transition_next_observed_lap"].sum()),
    )
    print(
        "Non-transition intervals:",
        int((result["transition_next_observed_lap"] == 0).sum()),
    )
    print(
        "Note: intervals are based on recorded lap rows. Missing lap records "
        "mean this is a next-observed-lap transition model, not an exact "
        "every-physical-lap hazard."
    )

    return result, feature_columns


def fit_discrete_hazard_model(lap_data, feature_columns):
    print("\n" + "=" * 75)
    print("MODEL B — LAP-LEVEL TRANSITION PROBABILITY")
    print("=" * 75)

    train, test = split_by_race(lap_data)

    categorical_features = [
        col for col in ["Compound", "Year"] if col in feature_columns
    ]
    numeric_features = [
        col for col in feature_columns if col not in categorical_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    classifier = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    X_train = train[feature_columns]
    y_train = train["transition_next_observed_lap"].astype(int)

    X_test = test[feature_columns]
    y_test = test["transition_next_observed_lap"].astype(int)

    if y_train.nunique() < 2:
        raise RuntimeError(
            "Training split contains only one outcome class. "
            "Cannot fit logistic regression."
        )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nDISCRETE-TIME MODEL HOLDOUT PERFORMANCE")
    print("-" * 75)
    print(f"Training lap rows: {len(train):,}")
    print(f"Testing lap rows:  {len(test):,}")
    print(f"Training races: {train['Race'].nunique()}")
    print(f"Testing races:  {test['Race'].nunique()}")
    print(f"Positive test intervals: {int(y_test.sum()):,}")
    print(f"Negative test intervals: {int((y_test == 0).sum()):,}")

    if y_test.nunique() == 2:
        auc = roc_auc_score(y_test, probabilities)
        average_precision = average_precision_score(y_test, probabilities)
    else:
        auc = np.nan
        average_precision = np.nan
        print(
            "Warning: test split contains only one outcome class; "
            "ROC AUC and average precision cannot be calculated."
        )

    brier = brier_score_loss(y_test, probabilities)

    print(f"ROC AUC: {auc:.4f}" if np.isfinite(auc) else "ROC AUC: N/A")
    print(
        f"Average precision: {average_precision:.4f}"
        if np.isfinite(average_precision)
        else "Average precision: N/A"
    )
    print(f"Brier score: {brier:.5f} (lower is better)")

    print("\nEXAMPLE NEXT-OBSERVED-LAP TRANSITION PROBABILITIES")
    examples = test[["Race", "Driver", "Compound", "LapNumber", "TyreAge"]].copy()
    examples["TransitionProbability"] = probabilities
    examples["ActualTransition"] = y_test.to_numpy()

    print(
        examples.head(12).to_string(
            index=False,
            formatters={
                "TransitionProbability": "{:.3%}".format,
            },
        )
    )

    return model, train, test, auc, average_precision, brier


# =========================================================
# MAIN
# =========================================================


def main(raw_data=None):
    print("\n" + "=" * 75)
    print("F1 TYRE STRATEGY — TWO-MODEL SURVIVAL PIPELINE")
    print("Local historical Parquet data | Offline")
    print("=" * 75)

    # 1. Load and prepare local lap records.
    raw = load_training_data() if raw_data is None else raw_data.copy()
    laps = prepare_laps(raw)

    # 2. Build stint-level observations and censoring labels.
    stints = build_stint_observations(laps)
    stints = add_censoring(stints)

    # 3. Keep fully dry HARD / MEDIUM / SOFT stints.
    dry_stints = filter_dry_stints(stints)

    # 4. Kaplan-Meier descriptive baseline.
    km_summary, km_models = fit_kaplan_meier(dry_stints)

    # 5. Model A: Weibull AFT stint-duration model.
    aft_results = fit_weibull_aft(dry_stints)

    # 6. Model B: lap-level discrete-time transition model.
    lap_hazard_data, hazard_features = build_lap_level_hazard_data(
        laps,
        dry_stints,
    )

    hazard_results = fit_discrete_hazard_model(
        lap_hazard_data,
        hazard_features,
    )

    # 7. Final summary.
    print("\n" + "=" * 75)
    print("FINAL DATA SUMMARY")
    print("=" * 75)
    actual_years = sorted(dry_stints["Year"].dropna().astype(int).unique().tolist())
    print(f"Seasons: {actual_years}")
    print(f"Races: {dry_stints['Race'].nunique()}")
    print(f"Dry stints: {len(dry_stints):,}")
    print(f"Lap-level observations: {len(lap_hazard_data):,}")
    print(
        "Stints with missing recorded laps:",
        int((dry_stints["missing_lap_count"] > 0).sum()),
    )
    print("Censoring reasons:")
    print(dry_stints["censoring_reason"].value_counts().to_string())

    print("\nMODELS COMPLETED")
    print("Model A: Weibull AFT — stint duration")
    print("Model B: Logistic hazard — next-observed-lap transition")
    print("Kaplan-Meier: descriptive survival baseline")
    print("\nNo data downloaded.")
    print("No CSV files or plots created.")
    print("=" * 75)

    return {
        "raw_laps": raw,
        "prepared_laps": laps,
        "stints": dry_stints,
        "km_summary": km_summary,
        "km_models": km_models,
        "aft_results": aft_results,
        "lap_hazard_data": lap_hazard_data,
        "hazard_results": hazard_results,
    }


# =========================================================
# EXECUTION
# =========================================================

if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    RESULTS = main()
