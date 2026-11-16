"""
F1 TYRE STRATEGY — 2026 HYBRID STRATEGY RECOMMENDER

Uses:
1. The project's HuberRegressor pace-loss model and exact features.
2. The 2022–2025 Weibull AFT and Kaplan–Meier survival data.
3. The lap-level logistic transition model.
4. Exhaustive one-stop and two-stop strategy comparison.

Run:
    uv run python recommend.py

With a FastF1-style race CSV:
    uv run python recommend.py --csv data/2026_race.csv --driver VER --total-laps 57

Optional:
    --lap 25
    --compound MEDIUM
    --tyre-age 12
    --rain
    --pit-loss 22
    --hard-code C1 --medium-code C3 --soft-code C5

The 2026 C-code allocation is event-specific. Supply the actual weekend
allocation using --hard-code / --medium-code / --soft-code. The defaults
are display labels only and must not be treated as an official allocation.

The script reads local Parquet training data and an optional race CSV.
It creates only one timestamped TXT report under ./reports/.
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_pace_loss import load_training_data as load_pace_training_data
from train_pace_loss import train_pace_loss, prepare_data as prepare_pace_data

from build_stints import (
    main as build_survival_data,
    prepare_aft_data,
    build_lap_level_hazard_data,
    WEATHER_COLUMNS,
)

from lifelines import WeibullAFTFitter


# ============================================================
# SETTINGS
# ============================================================

HERE = Path(__file__).resolve().parent
REPORT_DIR = HERE / "reports"

DRY = ("HARD", "MEDIUM", "SOFT")
MIN_STINT_LAPS = 5
MAX_STINT_LAPS = 45
DEFAULT_PIT_LOSS = 22.0
TOP_STRATEGIES = 8
REQUIRE_TWO_COMPOUNDS = True

# Pace model was trained on stint-relative age and predicts from
# approximately stint lap 10 onward. Do not extrapolate beyond the
# 98th percentile age observed in its training data.
PACE_MODEL_START_AGE = 9


# ============================================================
# GENERAL HELPERS
# ============================================================


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "lapnumber": "LapNumber",
        "totallaps": "TotalLaps",
        "total_laps": "TotalLaps",
        "driver": "Driver",
        "drivercode": "Driver",
        "compound": "Compound",
        "tyrelife": "TyreLife",
        "stint": "Stint",
        "tracktemp": "TrackTemp",
        "airtemp": "AirTemp",
        "humidity": "Humidity",
        "pressure": "Pressure",
        "windspeed": "WindSpeed",
        "rainfall": "Rainfall",
        "freshtyre": "FreshTyre",
        "ccode": "CCode",
    }
    rename = {}
    for column in df.columns:
        key = str(column).strip().replace(" ", "").replace("-", "").lower()
        if key in aliases:
            rename[column] = aliases[key]
    return df.rename(columns=rename)


def is_rain(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "rain", "wet"}


def safe_float(value, default=np.nan) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default




def race_pace_feature_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create compound-specific tyre-age features for race-normalised lap pace."""
    frame = data.copy()
    frame["TyreLife"] = pd.to_numeric(frame["TyreLife"], errors="coerce")
    frame["LapNumber"] = pd.to_numeric(frame["LapNumber"], errors="coerce")
    race_max = frame.groupby("Race")["LapNumber"].transform("max")
    frame["RaceProgress"] = (frame["LapNumber"] / race_max).clip(0.0, 1.0)

    features = pd.DataFrame(index=frame.index)
    compounds = ("HARD", "MEDIUM", "SOFT")
    for compound in compounds:
        mask = frame["Compound"].astype(str).str.upper().eq(compound).astype(float)
        age = frame["TyreLife"].clip(lower=0.0)
        features[f"is_{compound}"] = mask
        features[f"age_{compound}"] = mask * age
        features[f"age2_{compound}"] = mask * age.pow(2) / 100.0

    features["RaceProgress"] = frame["RaceProgress"]
    features["RaceProgress2"] = frame["RaceProgress"] ** 2

    weather_columns = [c for c in WEATHER_COLUMNS if c in frame.columns]
    for col in weather_columns:
        features[col] = pd.to_numeric(frame[col], errors="coerce")

    return features.astype(float), list(features.columns)


def fit_race_normalised_pace_model(training_laps: pd.DataFrame) -> dict:
    """
    Estimate lap-time differences relative to each historical race-driver median.

    The common race-driver pace level cancels when comparing strategies for the
    same driver and race. This is a relative pace model, not an absolute lap-time
    or finishing-position model.
    """
    clean = prepare_pace_data(training_laps)
    required = ["LapTimeSeconds", "Race", "Driver", "Compound", "TyreLife", "LapNumber"]
    clean = clean.dropna(subset=required).copy()
    clean["LapTimeSeconds"] = pd.to_numeric(clean["LapTimeSeconds"], errors="coerce")
    clean = clean[clean["LapTimeSeconds"].between(30, 300)].copy()

    if clean.empty:
        raise ValueError("No usable dry lap times available for race-pace calibration.")

    group = clean.groupby(["Race", "Driver"])["LapTimeSeconds"]
    clean["RaceDriverBaseline"] = group.transform("median")
    clean["RelativeLapTime"] = clean["LapTimeSeconds"] - clean["RaceDriverBaseline"]

    X, feature_columns = race_pace_feature_frame(clean)
    y = clean["RelativeLapTime"].astype(float)

    medians = X.median().fillna(0.0)
    X = X.fillna(medians)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("huber", HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500)),
    ])
    model.fit(X, y)

    supported_ages = {}
    for compound in DRY:
        ages = clean.loc[clean["Compound"].astype(str).str.upper() == compound, "TyreLife"]
        if not ages.empty:
            supported_ages[compound] = float(ages.quantile(0.98))

    print("\nRACE-NORMALISED LAP-PACE CALIBRATION")
    print("-" * 72)
    print(f"Training lap rows: {len(clean):,}")
    print(f"Training races: {clean['Race'].nunique()}")
    print(f"Training seasons: {sorted(clean['Year'].dropna().astype(int).unique().tolist())}")
    print("Target: lap time relative to the same driver's median in each race.")
    print("This estimates comparative tyre pace, not absolute race time.")

    return {
        "model": model,
        "feature_columns": feature_columns,
        "imputation_values": medians.to_dict(),
        "supported_ages": supported_ages,
        "training_years": sorted(clean["Year"].dropna().astype(int).unique().tolist()),
        "training_rows": len(clean),
    }


def predict_relative_race_pace(
    models, compound: str, tyre_age: int, race_progress: float, weather: dict
) -> float:
    pace = models["race_pace"]
    if compound not in pace["supported_ages"]:
        raise ValueError(f"No race-pace calibration for {compound}.")
    supported = pace["supported_ages"][compound]
    if tyre_age > supported:
        raise ValueError(
            f"{compound} tyre age {tyre_age} exceeds race-pace support "
            f"({supported:.1f} laps)."
        )

    row = {name: 0.0 for name in pace["feature_columns"]}
    row[f"is_{compound}"] = 1.0
    row[f"age_{compound}"] = float(tyre_age)
    row[f"age2_{compound}"] = float(tyre_age ** 2) / 100.0
    row["RaceProgress"] = float(np.clip(race_progress, 0.0, 1.0))
    row["RaceProgress2"] = row["RaceProgress"] ** 2
    for col in WEATHER_COLUMNS:
        if col in row:
            row[col] = safe_float(
                weather.get(col), pace["imputation_values"].get(col, 0.0)
            )
    frame = pd.DataFrame([row], columns=pace["feature_columns"])
    return float(pace["model"].predict(frame)[0])



# ============================================================
# LOAD AND REFIT MODELS ON ALL AVAILABLE HISTORICAL DATA
# ============================================================


def fit_hazard_full(lap_data: pd.DataFrame, feature_columns: list[str]):
    categorical = [c for c in ("Compound", "Year") if c in feature_columns]
    numeric = [c for c in feature_columns if c not in categorical]

    transformers = []
    if numeric:
        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipe, numeric))

    if categorical:
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipe, categorical))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )

    X = lap_data[feature_columns].copy()
    y = lap_data["transition_next_observed_lap"].astype(int)

    if y.nunique() < 2:
        raise RuntimeError("Historical transition data contains only one class.")

    model.fit(X, y)
    return model


def load_models(training_laps: pd.DataFrame | None = None):
    print("\n" + "=" * 72)
    print("F1 TYRE STRATEGY — FITTING HUBER-BASED RECOMMENDATION MODELS")
    print("=" * 72)

    print("\n[1/3] Training Huber pace-loss model from local Parquet data")
    pace_laps = (
        load_pace_training_data() if training_laps is None else training_laps.copy()
    )
    if training_laps is not None:
        print(
            "Walk-forward validation mode: target race and target season are excluded."
        )
    pace_result = train_pace_loss(pace_laps)
    race_pace = fit_race_normalised_pace_model(pace_laps)

    pace_model = pace_result["model"]
    pace_features = pace_result["feature_columns"]
    pace_compounds = pace_result["compounds"]
    supported_ages = pace_result["supported_ages"]

    if pace_model is None or not pace_features:
        raise RuntimeError("Pace-loss model or feature information is missing.")

    print("\n[2/3] Building survival and transition datasets")
    survival = build_survival_data(raw_data=pace_laps)

    if not isinstance(survival, dict):
        raise TypeError(
            "build_stints.main() did not return the expected result dictionary."
        )

    stints = survival["stints"]
    km_summary = survival["km_summary"]
    lap_hazard_data = survival["lap_hazard_data"]

    # Refit the Weibull AFT model on all available historical stints.
    # The holdout metrics printed by build_stints.py remain the reported
    # independent evaluation; this refit is for recommendation use only.
    aft_data, aft_numeric = prepare_aft_data(stints)
    formula = " + ".join(["C(Compound)", "C(Year)", *aft_numeric])
    aft_model = WeibullAFTFitter(penalizer=0.01)
    aft_model.fit(
        aft_data,
        duration_col="duration",
        event_col="event",
        formula=formula,
    )

    print("\n[3/3] Fitting lap-level transition model on all historical rows")
    # Reconstruct feature names exactly as the survival pipeline does.
    hazard_features = [
        "Compound",
        "Year",
        "TyreAge",
        "RaceProgress",
        *[c for c in WEATHER_COLUMNS if c in lap_hazard_data.columns],
    ]
    hazard_model = fit_hazard_full(lap_hazard_data, hazard_features)

    medians = {}
    for _, row in km_summary.iterrows():
        compound = str(row["Compound"]).upper()
        median = safe_float(row.get("KM median"))
        if compound in DRY and np.isfinite(median):
            medians[compound] = median

    return {
        "pace_model": pace_model,
        "race_pace": race_pace,
        "pace_features": list(pace_features),
        "pace_compounds": list(pace_compounds),
        "supported_ages": supported_ages,
        "pace_training": pace_result["training_data"],
        "aft_model": aft_model,
        "aft_numeric": aft_numeric,
        "aft_training": aft_data,
        # Training-derived medians for prediction-time missing weather values.
        "aft_imputation_values": {
            col: float(pd.to_numeric(aft_data[col], errors="coerce").median())
            if pd.to_numeric(aft_data[col], errors="coerce").notna().any()
            else 0.0
            for col in aft_numeric
        },
        "hazard_model": hazard_model,
        "hazard_features": hazard_features,
        "hazard_training": lap_hazard_data,
        "hazard_supported_ages": {
            str(compound).upper(): float(group["TyreAge"].quantile(0.98))
            for compound, group in lap_hazard_data.groupby("Compound")
            if pd.to_numeric(group["TyreAge"], errors="coerce").notna().any()
        },
        "aft_supported_durations": {
            str(compound).upper(): float(group["duration"].quantile(0.98))
            for compound, group in aft_data.groupby("Compound")
            if pd.to_numeric(group["duration"], errors="coerce").notna().any()
        },
        "km_medians": medians,
        "km_summary": km_summary,
    }


# ============================================================
# PACE-LOSS MODEL
# ============================================================


def pace_features_for_age(
    models, compound: str, relative_age: float, race_progress: float
) -> pd.DataFrame:
    row = {name: 0.0 for name in models["pace_features"]}

    age_name = f"age_{compound}"
    age2_name = f"age2_{compound}"

    if age_name not in row or age2_name not in row:
        raise ValueError(f"Missing expected pace features for {compound}.")

    row[age_name] = float(relative_age)
    row[age2_name] = float(relative_age**2) / 100.0

    if "RaceProgress" in row:
        row["RaceProgress"] = float(np.clip(race_progress, 0.0, 1.0))
    if "RaceProgress2" in row:
        row["RaceProgress2"] = float(np.clip(race_progress, 0.0, 1.0) ** 2)

    return pd.DataFrame([row], columns=models["pace_features"])


def predict_pace_loss(
    models, compound: str, relative_age: int, race_progress: float
) -> float:
    if compound not in models["pace_compounds"]:
        raise ValueError(f"{compound} is unavailable in the trained pace model.")

    supported = float(models["supported_ages"].get(compound, np.nan))
    if not np.isfinite(supported) or relative_age > supported:
        raise ValueError(
            f"{compound} relative tyre age {relative_age} exceeds "
            f"the supported training range ({supported:.1f})."
        )

    if relative_age < PACE_MODEL_START_AGE:
        return 0.0

    X = pace_features_for_age(models, compound, relative_age, race_progress)
    value = float(models["pace_model"].predict(X)[0])
    if not np.isfinite(value):
        raise RuntimeError("Pace model returned a non-finite prediction.")
    return value


# ============================================================
# SURVIVAL AND TRANSITION PREDICTIONS
# ============================================================


def _aft_prediction_row(
    models,
    compound: str,
    year: int,
    starting_age: float,
    race_progress: float,
    weather: dict,
) -> pd.DataFrame:
    row = {
        "Compound": compound,
        "Year": str(year),
        "starting_tyre_age": starting_age,
        "RaceProgress": race_progress,
        "RaceProgressSquared": race_progress**2,
    }
    imputation_values = models.get("aft_imputation_values", {})
    for col in models["aft_numeric"]:
        if col not in row:
            value = safe_float(weather.get(col), np.nan)
            if not np.isfinite(value):
                value = safe_float(imputation_values.get(col), 0.0)
            row[col] = value

    frame = pd.DataFrame([row])
    for col in models["aft_numeric"]:
        if col not in frame:
            frame[col] = safe_float(imputation_values.get(col), 0.0)
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame[col] = frame[col].fillna(safe_float(imputation_values.get(col), 0.0))
    return frame[["Compound", "Year", *models["aft_numeric"]]]


def predict_aft_median(
    models,
    compound: str,
    year: int,
    starting_age: float,
    race_progress: float,
    weather: dict,
) -> float:
    frame = _aft_prediction_row(
        models, compound, year, starting_age, race_progress, weather
    )
    prediction = models["aft_model"].predict_median(frame)
    value = float(np.asarray(prediction).reshape(-1)[0])
    return value if np.isfinite(value) else np.nan


def predict_aft_survival(
    models,
    compound: str,
    year: int,
    starting_age: float,
    race_progress: float,
    weather: dict,
    duration: int,
) -> float:
    """Probability a fresh historical-style stint lasts beyond duration."""
    frame = _aft_prediction_row(
        models, compound, year, starting_age, race_progress, weather
    )
    time_point = max(int(duration), 1)
    survival = models["aft_model"].predict_survival_function(frame, times=[time_point])
    value = float(survival.iloc[0, 0])
    return float(np.clip(value, 0.0, 1.0))


def predict_transition_probability(
    models,
    compound: str,
    year: int,
    tyre_age: float,
    race_progress: float,
    weather: dict,
) -> float:
    row = {
        "Compound": compound,
        "Year": str(year),
        "TyreAge": tyre_age,
        "RaceProgress": race_progress,
    }
    for col in models["hazard_features"]:
        if col not in row:
            row[col] = safe_float(weather.get(col), np.nan)

    frame = pd.DataFrame([row], columns=models["hazard_features"])
    probability = float(models["hazard_model"].predict_proba(frame)[0, 1])
    return float(np.clip(probability, 0.0, 1.0))


# ============================================================
# 2026 RACE INPUT
# ============================================================


def load_race_state(args, all_laps: pd.DataFrame | None = None) -> dict:
    if args.race_year is not None or args.race_event is not None:
        if all_laps is None:
            raise ValueError("Internal error: Parquet race data was not loaded.")
        if args.race_year is None or not args.race_event:
            raise ValueError("Use both --race-year and --race-event.")

        data = all_laps.copy()
        data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
        data["LapNumber"] = pd.to_numeric(data["LapNumber"], errors="coerce")
        data["TyreLife"] = pd.to_numeric(data["TyreLife"], errors="coerce")
        event_mask = (
            data["Event"]
            .astype(str)
            .str.contains(args.race_event, case=False, regex=False, na=False)
        )
        race_data = data[(data["Year"] == args.race_year) & event_mask].copy()
        if race_data.empty:
            available = sorted(
                data.loc[data["Year"] == args.race_year, "Event"]
                .dropna()
                .astype(str)
                .unique()
            )
            raise ValueError(
                f"No event matched {args.race_event!r} in {args.race_year}. "
                f"Available events: {', '.join(available)}"
            )

        race_data = race_data.dropna(
            subset=["LapNumber", "Driver", "Compound", "TyreLife"]
        )
        if args.driver:
            driver_data = race_data[
                race_data["Driver"].astype(str).str.upper() == args.driver.upper()
            ].copy()
            if driver_data.empty:
                drivers = sorted(race_data["Driver"].astype(str).str.upper().unique())
                raise ValueError(
                    f"Driver {args.driver} not found. Available drivers: {', '.join(drivers)}"
                )
        else:
            raise ValueError(
                "For Parquet race validation, specify --driver (e.g. --driver VER)."
            )

        total_laps = int(race_data["LapNumber"].max())
        current_lap = int(
            args.lap if args.lap is not None else driver_data["LapNumber"].max()
        )
        if current_lap < 1 or current_lap >= total_laps:
            raise ValueError(
                f"Choose --lap between 1 and {total_laps - 1} for an in-race recommendation."
            )
        observed = driver_data[driver_data["LapNumber"] <= current_lap].sort_values(
            "LapNumber"
        )
        if observed.empty:
            raise ValueError(
                f"No recorded laps for {args.driver} by lap {current_lap}."
            )
        latest = observed.iloc[-1]
        compound = str(args.compound or latest["Compound"]).upper().strip()
        if compound not in DRY:
            raise ValueError(
                f"Current compound {compound!r} is not a dry model compound."
            )
        tyre_age = int(
            args.tyre_age
            if args.tyre_age is not None
            else round(float(latest["TyreLife"]))
        )

        rainfall = bool(args.rain)
        if "Rainfall" in observed.columns:
            rainfall = rainfall or observed["Rainfall"].map(is_rain).any()
        weather = {}
        for col in WEATHER_COLUMNS:
            if col in observed.columns:
                values = pd.to_numeric(observed[col], errors="coerce").dropna()
                weather[col] = (
                    safe_float(values.iloc[-1]) if not values.empty else np.nan
                )

        training_years = sorted(
            pd.to_numeric(data["Year"], errors="coerce").dropna().astype(int).unique()
        )
        prior_years = [year for year in training_years if year < args.race_year]
        if not prior_years:
            raise ValueError(
                "No earlier seasons are available to train a leakage-free validation model."
            )

        actual_future_stints = []
        if "Stint" in driver_data.columns:
            future_rows = driver_data[driver_data["LapNumber"] > current_lap].copy()
            if not future_rows.empty:
                for stint_id, group in future_rows.groupby("Stint", sort=True):
                    group = group.sort_values("LapNumber")
                    actual_future_stints.append(
                        {
                            "stint": int(stint_id) if pd.notna(stint_id) else None,
                            "compound": str(group["Compound"].iloc[0]).upper(),
                            "first_lap": int(group["LapNumber"].min()),
                            "last_lap": int(group["LapNumber"].max()),
                            "laps_recorded": int(group["LapNumber"].nunique()),
                        }
                    )

        return {
            "driver": args.driver.upper(),
            "current_lap": current_lap,
            "total_laps": total_laps,
            "compound": compound,
            "tyre_age": tyre_age,
            "rainfall": rainfall,
            "weather": weather,
            "source": f"Local Parquet | {args.race_year} {latest['Event']} | walk-forward holdout",
            "year": args.race_year,
            "model_year": max(prior_years),
            "event": str(latest["Event"]),
            "training_years": prior_years,
            "actual_future_stints": actual_future_stints,
            "used_compounds": sorted(
                driver_data.loc[driver_data["LapNumber"] <= current_lap, "Compound"]
                .astype(str).str.upper().unique().tolist()
            ),
        }

    if args.csv is None:
        return {
            "driver": "DEMO DRIVER",
            "current_lap": args.lap or 25,
            "total_laps": args.total_laps or 57,
            "compound": args.compound or "MEDIUM",
            "tyre_age": args.tyre_age if args.tyre_age is not None else 12,
            "rainfall": bool(args.rain),
            "weather": {},
            "source": "Built-in demonstration scenario",
            "year": 2026,
            "model_year": 2025,
            "event": "Demonstration",
            "training_years": [2022, 2023, 2024, 2025],
            "used_compounds": [str(args.compound or "MEDIUM").upper()],
        }

    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(f"Race CSV not found: {path}")

    df = normalise_columns(pd.read_csv(path))
    required = ["LapNumber", "Compound", "TyreLife"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Race CSV is missing required columns: {missing}")

    if args.driver:
        if "Driver" not in df.columns:
            raise ValueError("CSV has no Driver column; cannot apply --driver.")
        selected = df[
            df["Driver"].astype(str).str.upper() == args.driver.upper()
        ].copy()
        if selected.empty:
            raise ValueError(f"Driver {args.driver} was not found in the CSV.")
    else:
        selected = df.copy()

    selected["LapNumber"] = pd.to_numeric(selected["LapNumber"], errors="coerce")
    selected["TyreLife"] = pd.to_numeric(selected["TyreLife"], errors="coerce")
    selected = selected.dropna(subset=["LapNumber", "Compound", "TyreLife"])
    if selected.empty:
        raise ValueError("No usable laps found for the selected driver.")

    latest = selected.sort_values("LapNumber").iloc[-1]
    current_lap = int(args.lap if args.lap is not None else latest["LapNumber"])
    if args.total_laps is not None:
        total_laps = int(args.total_laps)
    elif "TotalLaps" in selected.columns:
        total_values = pd.to_numeric(selected["TotalLaps"], errors="coerce").dropna()
        if total_values.empty:
            raise ValueError("TotalLaps column contains no valid values.")
        total_laps = int(total_values.max())
    else:
        raise ValueError("Race distance unknown. Supply --total-laps.")

    compound = str(args.compound or latest["Compound"]).upper().strip()
    tyre_age = int(
        args.tyre_age if args.tyre_age is not None else round(float(latest["TyreLife"]))
    )
    rainfall = bool(args.rain)
    if "Rainfall" in selected.columns:
        rainfall = rainfall or selected["Rainfall"].map(is_rain).any()
    weather = {}
    for col in WEATHER_COLUMNS:
        if col in selected.columns:
            values = pd.to_numeric(selected[col], errors="coerce").dropna()
            weather[col] = safe_float(values.iloc[-1]) if not values.empty else np.nan
    return {
        "driver": args.driver or str(latest.get("Driver", "Selected driver")),
        "current_lap": current_lap,
        "total_laps": total_laps,
        "compound": compound,
        "tyre_age": tyre_age,
        "rainfall": rainfall,
        "weather": weather,
        "source": str(path),
        "year": 2026,
        "model_year": 2025,
        "event": "CSV input",
        "training_years": [2022, 2023, 2024, 2025],
        "used_compounds": sorted(
            selected.loc[selected["LapNumber"] <= current_lap, "Compound"]
            .astype(str).str.upper().unique().tolist()
        ),
    }


# ============================================================
# STRATEGY GENERATION
# ============================================================


def stint_splits(total_laps: int, count: int):
    if count == 1:
        if 1 <= total_laps <= MAX_STINT_LAPS:
            yield (total_laps,)
        return

    if total_laps < count * MIN_STINT_LAPS:
        return
    if total_laps > count * MAX_STINT_LAPS:
        return

    if count == 2:
        for first in range(MIN_STINT_LAPS, MAX_STINT_LAPS + 1):
            second = total_laps - first
            if MIN_STINT_LAPS <= second <= MAX_STINT_LAPS:
                yield (first, second)
    elif count == 3:
        for first in range(MIN_STINT_LAPS, MAX_STINT_LAPS + 1):
            for second in range(MIN_STINT_LAPS, MAX_STINT_LAPS + 1):
                third = total_laps - first - second
                if MIN_STINT_LAPS <= third <= MAX_STINT_LAPS:
                    yield (first, second, third)


def current_stint_survival(models, race, compound: str, planned_laps: int) -> float:
    """Approximate probability of no observed transition before the planned stop.

    The hazard model predicts a transition on the next observed lap. We only
    accumulate the first planned_laps - 1 intervals so the intended stop at
    the segment boundary is not counted as an early transition.
    """
    survival = 1.0
    intervals = max(int(planned_laps) - 1, 0)
    for offset in range(intervals):
        age = race["tyre_age"] + offset
        progress = min((race["current_lap"] + offset) / race["total_laps"], 1.0)
        probability = predict_transition_probability(
            models, compound, race["model_year"], age, progress, race["weather"]
        )
        survival *= 1.0 - probability
    return float(np.clip(survival, 0.0, 1.0))


def generate_strategies(
    models, race, pit_loss: float, available_compounds: list[str]
) -> list[dict]:
    current = race["compound"]
    if current not in DRY:
        raise ValueError(
            f"Current compound must be one of {DRY}; got {current!r}. "
            "For C1-C5 inputs, map the weekend allocation to HARD/MEDIUM/SOFT."
        )

    remaining = race["total_laps"] - race["current_lap"]
    if remaining <= 0:
        return []
    if current not in available_compounds:
        raise ValueError(f"Current compound {current} is not in this weekend's allocation.")

    strategies = []
    compounds_already_used = {
        str(c).upper() for c in race.get("used_compounds", [current])
    }

    for n_stints in (1, 2, 3):
        if n_stints == 1:
            sequences = [(current,)]
        elif n_stints == 2:
            sequences = [(current, c) for c in available_compounds]
        else:
            sequences = [
                (current, a, b)
                for a, b in itertools.product(available_compounds, repeat=2)
            ]

        for lengths in stint_splits(remaining, n_stints):
            for sequence in sequences:
                compounds_in_race = compounds_already_used | set(sequence)
                if REQUIRE_TWO_COMPOUNDS and len(compounds_in_race) < 2:
                    continue

                details = []
                survival_values = []
                total_relative_pace = 0.0
                race_start = race["current_lap"]
                valid = True

                for index, (compound, length) in enumerate(zip(sequence, lengths)):
                    start_age = int(race["tyre_age"]) if index == 0 else 1

                    if index == 0:
                        hazard_limit = models.get("hazard_supported_ages", {}).get(compound)
                        planned_end_age = start_age + int(length) - 2
                        if hazard_limit is not None and planned_end_age > hazard_limit:
                            valid = False
                            break
                        survival = current_stint_survival(models, race, compound, length)
                        survival_source = "logistic next-observed-lap hazard"
                        aft_median = np.nan
                    else:
                        survival_duration = max(int(length) - 1, 1)
                        aft_limit = models.get("aft_supported_durations", {}).get(compound)
                        if aft_limit is not None and survival_duration > aft_limit:
                            valid = False
                            break
                        try:
                            survival = predict_aft_survival(
                                models, compound, race["model_year"], 0.0,
                                min(race_start / race["total_laps"], 1.0),
                                race["weather"], survival_duration,
                            )
                            aft_median = predict_aft_median(
                                models, compound, race["model_year"], 0.0,
                                min(race_start / race["total_laps"], 1.0),
                                race["weather"],
                            )
                        except Exception as exc:
                            valid = False
                            errors = models.setdefault("_aft_prediction_errors", {})
                            message = f"{type(exc).__name__}: {exc}"
                            errors[message] = errors.get(message, 0) + 1
                            break
                        survival_source = "Weibull AFT"

                    if not np.isfinite(survival):
                        valid = False
                        break

                    stint_relative_pace = 0.0
                    for offset in range(int(length)):
                        tyre_age = start_age + offset
                        progress = min((race_start + offset) / race["total_laps"], 1.0)
                        try:
                            lap_delta = predict_relative_race_pace(
                                models, compound, tyre_age, progress, race["weather"]
                            )
                        except Exception:
                            valid = False
                            break
                        stint_relative_pace += lap_delta

                    if not valid:
                        break

                    total_relative_pace += stint_relative_pace
                    km_median = models["km_medians"].get(compound, np.nan)
                    risk = 1.0 - float(np.clip(survival, 0.0, 1.0))
                    survival_values.append(max(float(survival), 1e-9))
                    details.append({
                        "compound": compound,
                        "length": int(length),
                        "start_age": int(start_age),
                        "end_age": int(start_age + length - 1),
                        "survival_probability": float(survival),
                        "early_transition_risk": risk,
                        "survival_source": survival_source,
                        "km_median": km_median,
                        "aft_median": aft_median,
                        "relative_pace_seconds": float(stint_relative_pace),
                        "warning": bool(
                            index > 0 and pd.notna(km_median) and length > 1.5 * km_median
                        ),
                    })
                    race_start += length

                if not valid:
                    continue

                stops = n_stints - 1
                combined_survival = float(np.prod(survival_values))
                combined_risk = float(np.clip(1.0 - combined_survival, 0.0, 1.0))
                scheduled_pit_loss = stops * pit_loss

                # If at least one planned stint ends prematurely, approximate
                # the consequence as one additional pit-loss event. This is an
                # explicit risk-cost proxy, not a calibrated retirement model.
                expected_unplanned_pit_loss = combined_risk * pit_loss
                estimated_race_time_cost = (
                    total_relative_pace + scheduled_pit_loss + expected_unplanned_pit_loss
                )

                strategies.append({
                    "stops": stops,
                    "stints": details,
                    "combined_survival": combined_survival,
                    "combined_early_transition_risk": combined_risk,
                    "relative_pace_seconds": float(total_relative_pace),
                    "pit_loss_assumption": float(scheduled_pit_loss),
                    "expected_unplanned_pit_loss": float(expected_unplanned_pit_loss),
                    "estimated_race_time_cost": float(estimated_race_time_cost),
                    "compounds_used": list(sequence),
                })

    strategies.sort(key=lambda item: (
        item["estimated_race_time_cost"],
        item["combined_early_transition_risk"],
        item["stops"],
    ))
    return strategies


def select_strategy_representatives(strategies: list[dict]) -> dict:
    """Select fastest, lowest-risk, and a transparent Pareto time/risk compromise."""
    if not strategies:
        return {}

    fastest = min(strategies, key=lambda item: item["estimated_race_time_cost"])
    lowest_risk = min(
        strategies,
        key=lambda item: (
            item["combined_early_transition_risk"],
            item["estimated_race_time_cost"],
        ),
    )

    # Keep only non-dominated candidates: no other candidate is both faster
    # (or equal) and lower-risk (or equal), with at least one strict improvement.
    pareto = []
    for candidate in strategies:
        dominated = any(
            other is not candidate
            and other["estimated_race_time_cost"] <= candidate["estimated_race_time_cost"]
            and other["combined_early_transition_risk"] <= candidate["combined_early_transition_risk"]
            and (
                other["estimated_race_time_cost"] < candidate["estimated_race_time_cost"]
                or other["combined_early_transition_risk"] < candidate["combined_early_transition_risk"]
            )
            for other in strategies
        )
        if not dominated:
            pareto.append(candidate)

    times = [item["estimated_race_time_cost"] for item in pareto]
    risks = [item["combined_early_transition_risk"] for item in pareto]
    time_span = max(max(times) - min(times), 1e-9)
    risk_span = max(max(risks) - min(risks), 1e-9)
    best_time = min(times)
    best_risk = min(risks)

    # Equal-normalised distance to the ideal (fastest, safest) point.
    balanced = min(
        pareto,
        key=lambda item: (
            ((item["estimated_race_time_cost"] - best_time) / time_span) ** 2
            + ((item["combined_early_transition_risk"] - best_risk) / risk_span) ** 2
        ),
    )
    return {"fastest": fastest, "lowest_risk": lowest_risk, "balanced": balanced}


# ============================================================
# REPORT
# ============================================================


def format_strategy(strategy: dict, index: int) -> str:
    lines = [
        f"\nOPTION {index} — {strategy['stops']}-STOP PLAN",
        "-" * 58,
    ]

    for number, stint in enumerate(strategy["stints"], start=1):
        lines.append(
            f"Stint {number}: {stint['compound']} | "
            f"{stint['length']} laps | tyre age {stint['start_age']}→{stint['end_age']}"
        )
        lines.append(
            f"  Estimated survival through planned stint: "
            f"{stint['survival_probability']:.1%}"
        )
        lines.append(
            f"  Early-transition risk estimate: {stint['early_transition_risk']:.1%} "
            f"({stint['survival_source']})"
        )
        if np.isfinite(stint["km_median"]):
            lines.append(f"  Historical KM median: {stint['km_median']:.1f} laps")
        if np.isfinite(stint["aft_median"]):
            lines.append(
                f"  Weibull AFT median estimate: {stint['aft_median']:.1f} laps"
            )
        if stint["warning"]:
            lines.append(
                "  CAUTION: planned fresh-tyre stint exceeds 1.5× the historical KM median."
            )

    lines.extend(
        [
            f"Stops: {strategy['stops']}",
            f"Approx. combined early-transition risk: "
            f"{strategy['combined_early_transition_risk']:.1%}",
            f"Approx. combined survival: {strategy['combined_survival']:.1%}",
            f"Relative tyre-pace contribution: {strategy['relative_pace_seconds']:+.1f}s",
            f"Scheduled pit loss: {strategy['pit_loss_assumption']:.1f}s",
            f"Expected extra pit-loss proxy from early-transition risk: "
            f"{strategy['expected_unplanned_pit_loss']:.1f}s",
            f"Estimated comparative time cost: {strategy['estimated_race_time_cost']:+.1f}s "
            "(lower is better; relative to race-driver baseline)",
        ]
    )
    return "\n".join(lines)


def create_report(models, race, strategies, args, transition_probability) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    path = REPORT_DIR / f"strategy_report_{now:%Y%m%d_%H%M%S}.txt"

    lines = [
        "=" * 72,
        "F1 TYRE STRATEGY — RACE-SPECIFIC STRATEGY REPORT",
        "=" * 72,
        f"Generated: {now:%Y-%m-%d %H:%M:%S}",
        f"Driver: {race['driver']}",
        f"Event: {race.get('event', 'Unknown')} ({race.get('year', 'Unknown')})",
        f"Training seasons: {race.get('training_years', 'not specified')}",
        f"Input: {race['source']}",
        f"Current lap: {race['current_lap']}/{race['total_laps']}",
        f"Remaining laps: {race['total_laps'] - race['current_lap']}",
        f"Current compound label: {race['compound']}",
        f"Current tyre age: {race['tyre_age']} laps",
        f"Rainfall detected: {'YES' if race['rainfall'] else 'NO'}",
        f"Pit-loss assumption: {args.pit_loss:.1f} seconds per stop",
        "",
        "2026 WEEKEND COMPOUND ALLOCATION",
        "-" * 72,
        f"Hard allocation:   {args.hard_code}",
        f"Medium allocation: {args.medium_code}",
        f"Soft allocation:   {args.soft_code}",
        "C-code values are user-supplied. NOT_SUPPLIED means no event allocation was entered.",
        "Verify the actual Pirelli allocation for this Grand Prix before using the mapping.",
        "",
        "CURRENT TRANSITION ESTIMATE",
        "-" * 72,
        (
            f"Next-observed-lap transition probability: {transition_probability:.2%}"
            if np.isfinite(transition_probability)
            else "Transition probability: unavailable"
        ),
        "This estimates an observed stint transition, not tyre failure or a guaranteed pit stop.",
        "",
        "MODEL NOTES",
        "-" * 72,
        f"Pace model: train_pace_loss.py, trained on seasons {race.get('training_years', 'not specified')}.",
        "Survival model: Weibull AFT, refitted on all available dry historical stints.",
        "Baseline: Kaplan–Meier compound-level survival estimates.",
        "Transition model: logistic probability of a next-observed-lap transition.",
        "The previously reported race-grouped holdout metrics remain the evaluation results.",
        "For Parquet race validation, all models exclude the target season and target race (walk-forward split).",
        "",
        "LIMITATIONS",
        "-" * 72,
        "The original Huber model still predicts deviation from an early-stint baseline.",
        "Strategy pace comparison uses a separate Huber model trained on lap time relative to each historical race-driver median.",
        "The common baseline cancels between strategies; the displayed time cost is comparative, not absolute finishing time.",
        "Strategies are ranked by comparative pace cost + scheduled pit loss + an expected extra-stop risk proxy.",
        "Same-compound consecutive stints are allowed when the race-level two-dry-compound requirement is already met.",
        "Pace, hazard-age, and AFT-duration predictions are restricted to their training-data support ranges.",
        "Combined early-transition risk multiplies stint survival estimates and assumes independence between stints.",
        "The risk-cost proxy assumes at most one additional pit-loss event if any planned stint transitions early.",
        "This is not a calibrated probability of winning or an absolute race-time forecast.",
        "It does not model traffic, track position, undercut/overcut, safety cars, fuel load,",
        "team-specific decisions, tyre warm-up, or the full 2026 car/tyre regulation changes.",
        "Historical HARD/MEDIUM/SOFT labels are relative weekend allocations, not fixed C-codes.",
        "The 2026 allocation mapping must be supplied and verified for each event.",
        "Wet-weather handling is rule-based; no wet-tyre ML model is available.",
        "",
    ]

    if race["rainfall"]:
        lines.extend(
            [
                "WET-WEATHER ROUTING",
                "-" * 72,
                "Rainfall detected. Dry-tyre strategy ranking has been skipped.",
                "INTERMEDIATE is shown as a rule-based indication only.",
                "Use the actual track condition and race-control information.",
                "",
            ]
        )
    elif not strategies:
        lines.append("No valid strategy could be generated for this race state.\n")
    else:
        lines.extend(["STRATEGY RANKING", "-" * 72])
        for i, strategy in enumerate(strategies[:TOP_STRATEGIES], start=1):
            lines.append(format_strategy(strategy, i))

        representatives = select_strategy_representatives(strategies)
        lines.extend(["", "STRATEGY PICKS", "-" * 72])
        for label, title in (
            ("fastest", "Fastest estimated comparative time"),
            ("lowest_risk", "Lowest estimated early-transition risk"),
            ("balanced", "Balanced time-risk compromise"),
        ):
            candidate = representatives[label]
            sequence = " -> ".join(
                f"{stint['compound']} ({stint['length']}L)"
                for stint in candidate["stints"]
            )
            lines.append(
                f"{title}: {sequence} | "
                f"time cost {candidate['estimated_race_time_cost']:+.1f}s | "
                f"risk {candidate['combined_early_transition_risk']:.1%}"
            )
        lines.append(
            "Balanced means the Pareto candidate closest to the ideal fastest-and-safest "
            "point after equal normalisation of time and risk; it is not a learned score."
        )

        best = representatives["fastest"]
        lines.extend(
            [
                "",
                "LOWEST ESTIMATED COMPARATIVE TIME-COST STRATEGY",
                "-" * 72,
                f"Stops: {best['stops']}",
                f"Comparative tyre-pace contribution: {best['relative_pace_seconds']:+.1f}s",
                f"Scheduled pit loss: {best['pit_loss_assumption']:.1f}s",
                f"Expected extra pit-loss proxy: {best['expected_unplanned_pit_loss']:.1f}s",
                f"Estimated comparative time cost: {best['estimated_race_time_cost']:+.1f}s",
                f"Approx. combined early-transition risk: {best['combined_early_transition_risk']:.1%}",
                "Lower comparative time cost is preferred; risk is included as an expected pit-loss proxy.",
                "This is not a guaranteed win prediction or an absolute finishing-time estimate.",
                "",
            ]
        )

    if race.get("actual_future_stints") is not None:
        lines.extend(
            [
                "",
                "POST-RACE VALIDATION REFERENCE (ACTUAL DATA AFTER THE SELECTED LAP)",
                "-" * 72,
            ]
        )
        actual = race["actual_future_stints"]
        if actual:
            for item in actual:
                lines.append(
                    f"Actual stint {item['stint']}: {item['compound']} | "
                    f"recorded laps {item['first_lap']}–{item['last_lap']} "
                    f"({item['laps_recorded']} rows)"
                )
            lines.append(
                "This is the observed post-decision outcome for comparison; it was not used to train the models."
            )
        else:
            lines.append(
                "No later tyre stint was recorded for this driver after the selected lap."
            )

    lines.extend(["=" * 72, "END OF REPORT", "=" * 72])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ============================================================
# DISPLAY AND CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="F1 tyre strategy recommender with historical race validation"
    )
    parser.add_argument("--csv", type=str, default=None, help="FastF1-style race CSV")
    parser.add_argument(
        "--race-year",
        type=int,
        default=None,
        help="Target season from local Parquet data",
    )
    parser.add_argument(
        "--race-event",
        type=str,
        default=None,
        help="Event name or unique substring from local Parquet data",
    )
    parser.add_argument(
        "--driver", type=str, default=None, help="Driver code, e.g. VER"
    )
    parser.add_argument("--total-laps", type=int, default=None, help="Total race laps")
    parser.add_argument("--lap", type=int, default=None, help="Override current lap")
    parser.add_argument(
        "--compound", choices=DRY, default=None, help="Current label: HARD/MEDIUM/SOFT"
    )
    parser.add_argument("--tyre-age", type=int, default=None, help="Current tyre age")
    parser.add_argument("--rain", action="store_true", help="Force wet-weather routing")
    parser.add_argument("--pit-loss", type=float, default=DEFAULT_PIT_LOSS)
    parser.add_argument(
        "--hard-code", default="NOT_SUPPLIED", help="2026 GP's nominated Hard C-code"
    )
    parser.add_argument(
        "--medium-code",
        default="NOT_SUPPLIED",
        help="2026 GP's nominated Medium C-code",
    )
    parser.add_argument(
        "--soft-code", default="NOT_SUPPLIED", help="2026 GP's nominated Soft C-code"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.pit_loss <= 0:
        raise ValueError("--pit-loss must be greater than zero.")

    all_laps = None
    if args.race_year is not None or args.race_event is not None:
        print("Loading local Parquet data to select a historical race...")
        all_laps = load_pace_training_data()
        if args.race_year is None or not args.race_event:
            raise ValueError("Use both --race-year and --race-event.")
        if args.race_year not in sorted(
            pd.to_numeric(all_laps["Year"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        ):
            raise ValueError(
                "Requested season is not present in the local Parquet dataset."
            )
        if args.race_event.lower() in {"list", "--list"}:
            events = (
                all_laps.loc[all_laps["Year"] == args.race_year, "Event"]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .sort_values()
            )
            print(f"\nAvailable {args.race_year} events:")
            for event in events:
                print(f"  {event}")
            return

    race = load_race_state(args, all_laps=all_laps)
    
    if race["current_lap"] < 1 or race["total_laps"] <= race["current_lap"]:
        raise ValueError(
            "Race lap values are invalid; total laps must exceed current lap."
        )
    if race["tyre_age"] < 0:
        raise ValueError("Tyre age cannot be negative.")

    if all_laps is not None:
        training_laps = all_laps[
            pd.to_numeric(all_laps["Year"], errors="coerce") < race["year"]
        ].copy()
        actual_training_years = sorted(
            pd.to_numeric(training_laps["Year"], errors="coerce")
            .dropna().astype(int).unique().tolist()
        )
        race["training_years"] = actual_training_years
        if training_laps.empty:
            raise ValueError(
                "No earlier-season training data remains after excluding the target season."
            )
    else:
        training_laps = None
    models = load_models(training_laps=training_laps)

    transition_probability = np.nan
    try:
        transition_probability = predict_transition_probability(
            models,
            race["compound"],
            race["model_year"],
            race["tyre_age"],
            race["current_lap"] / race["total_laps"],
            race["weather"],
        )
    except Exception as exc:
        warnings.warn(f"Transition estimate unavailable: {exc}")

    strategies = []
    if not race["rainfall"]:
        strategies = generate_strategies(
            models,
            race,
            args.pit_loss,
            list(models["pace_compounds"]),
        )

    aft_errors = models.get("_aft_prediction_errors", {})
    if aft_errors:
        print("\\nAFT prediction warnings (affected candidates were skipped):")
        for message, count in sorted(aft_errors.items(), key=lambda item: -item[1]):
            print(f"  {count} candidate(s): {message}")

    report = create_report(
        models,
        race,
        strategies,
        args,
        transition_probability,
    )

    print("\n" + "=" * 72)
    print("F1 TYRE STRATEGY — RECOMMENDATIONS")
    print("=" * 72)
    print(f"Driver: {race['driver']}")
    print(f"Event: {race.get('event', 'Unknown')} ({race.get('year', 'Unknown')})")
    print(f"Race: {race['current_lap']}/{race['total_laps']} laps")
    print(f"Current tyre: {race['compound']} | age {race['tyre_age']}")
    print(
        f"Transition probability: {transition_probability:.2%}"
        if np.isfinite(transition_probability)
        else "Transition probability: unavailable"
    )

    if race["rainfall"]:
        print("\nRain detected: dry strategy comparison skipped.")
        print("Rule-based indication: INTERMEDIATE")
    elif not strategies:
        print("\nNo valid strategy generated.")
    else:
        print("\nTOP STRATEGIES (ranked by estimated comparative race-time cost)")
        for i, strategy in enumerate(strategies[:TOP_STRATEGIES], start=1):
            sequence = " -> ".join(
                f"{stint['compound']} ({stint['length']}L)"
                for stint in strategy["stints"]
            )
            print(
                f"{i:>2}. {strategy['stops']}-stop | {sequence} | "
                f"comparative time cost {strategy['estimated_race_time_cost']:+.1f}s | "
                f"risk {strategy['combined_early_transition_risk']:.1%} | "
                f"scheduled pit loss {strategy['pit_loss_assumption']:.1f}s"
            )

        print("\nBEST CANDIDATE BY STOP COUNT")
        for stop_count in sorted({item["stops"] for item in strategies}):
            candidate = next(item for item in strategies if item["stops"] == stop_count)
            sequence = " -> ".join(
                f"{stint['compound']} ({stint['length']}L)"
                for stint in candidate["stints"]
            )
            print(
                f"{stop_count}-stop | {sequence} | "
                f"comparative time cost {candidate['estimated_race_time_cost']:+.1f}s | "
                f"risk {candidate['combined_early_transition_risk']:.1%} | "
                f"scheduled pit loss {candidate['pit_loss_assumption']:.1f}s"
            )

        print(
            "\nNOTE: Ranking uses relative tyre pace + scheduled pit loss + an approximate "
            "risk-based extra-stop cost. It is not an absolute race-time or win prediction."
        )
    actual_future = race.get("actual_future_stints")
    if actual_future is not None:
        print("\nACTUAL POST-DECISION STINTS (validation reference)")
        if actual_future:
            for item in actual_future:
                print(
                    f"  {item['compound']}: laps {item['first_lap']}–{item['last_lap']} "
                    f"({item['laps_recorded']} recorded laps)"
                )
        else:
            print("  No later stint recorded.")

    print(f"\nReport saved: {report}")
    print("=" * 72)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
