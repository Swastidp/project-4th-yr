"""Stage-1 pace-loss model: predict lap delta vs the driver's own best lap in that race.

Two models on the same target, per the design doc:
  - Ridge with an explicit per-compound spec, for quotable coefficients.
  - XGBoost, for accuracy.

Both scored with GroupKFold by race (Year+Event) so no race appears in train and test.
Day 1's chart showed HARD/MEDIUM getting *faster* with age; that is fuel burn, not physics.
FuelPct = 1 - LapNumber/TotalLaps is the correction.

Run:  uv run train_pace_loss.py
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
IN_CSV = DATA_DIR / "laps_clean.csv"
OUT_PNG = DATA_DIR / "degradation_curve_corrected.png"

MIN_LAPS = 20  # same cutoff plot_degradation.py uses for the raw-means panel
XGB_FEATURES = ["TyreLife", "FuelPct", "TrackTemp", "AirTemp", "Humidity"]
# Event (race name) is deliberately NOT an XGBoost feature. Under GroupKFold the held-out
# race has every Event column at zero — a region the trees never saw, and trees can't
# extrapolate there. With Event in, XGBoost lost to the median baseline (2.284 vs 2.262).
WET_FRACTION = 0.10  # a race with rain on >10% of its clean laps is a separate regime (doc Part 4)


def add_features(df):
    """Target + fuel load. Delta is per driver per race; FuelPct is per race."""
    best = df.groupby(["Year", "Event", "Driver"])["LapTimeSeconds"].transform("min")
    df["Delta"] = df["LapTimeSeconds"] - best

    # ponytail: TotalLaps = each race's max LapNumber in the *cleaned* CSV, not
    # session.total_laps. Exact for the dry races (their last lap survives the filters);
    # a race that loses its final lap to the filters reads one short — swap in FastF1's
    # session.total_laps if that ever matters.
    total = df.groupby(["Year", "Event"])["LapNumber"].transform("max")
    df["FuelPct"] = 1 - df["LapNumber"] / total

    df["Race"] = df["Year"].astype(str) + " " + df["Event"]
    return df


def ridge_design(df, compounds):
    """Per-compound intercept + linear and quadratic age, plus shared fuel and track temp.

    Delta = b_c + b_c1*age + b_c2*age^2/100 + g*FuelPct + d*TrackTemp
    Explicit dummy/interaction columns so each compound's age coefficient is readable after.
    """
    X = pd.DataFrame(index=df.index)
    for c in compounds:
        is_c = (df["Compound"] == c).astype(float)
        X[f"int[{c}]"] = is_c
        X[f"age[{c}]"] = is_c * df["TyreLife"]
        X[f"age2[{c}]"] = is_c * df["TyreLife"] ** 2 / 100
    X["FuelPct"] = df["FuelPct"]
    X["TrackTemp"] = df["TrackTemp"]
    return X


def make_ridge():
    # No intercept: the per-compound dummies are the intercepts, which keeps them quotable.
    return Ridge(alpha=1.0, fit_intercept=False)


def make_xgb():
    return XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=4,
    )


def cv_scores(X, y, groups, make_model):
    """One held-out race per fold. Returns per-fold (race, mae, r2) plus baseline maes."""
    rows = []
    for train_idx, test_idx in GroupKFold(n_splits=groups.nunique()).split(X, y, groups):
        model = make_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        base = np.full(len(test_idx), y.iloc[train_idx].median())
        rows.append({
            "race": groups.iloc[test_idx].iloc[0],
            "mae": mean_absolute_error(y.iloc[test_idx], pred),
            "r2": r2_score(y.iloc[test_idx], pred),
            "base_mae": mean_absolute_error(y.iloc[test_idx], base),
            "base_r2": r2_score(y.iloc[test_idx], base),
        })
    return pd.DataFrame(rows)


def print_folds(name, folds):
    print(f"\n--- {name} (GroupKFold, one whole race held out per fold) ---")
    print(f"{'held-out race':<28}{'MAE':>8}{'R2':>8}{'base MAE':>10}{'base R2':>9}{'beats?':>8}")
    for _, r in folds.iterrows():
        win = "yes" if r["mae"] < r["base_mae"] else "NO"
        print(f"{r['race']:<28}{r['mae']:>8.3f}{r['r2']:>8.3f}{r['base_mae']:>10.3f}{r['base_r2']:>9.3f}{win:>8}")
    m = folds.mean(numeric_only=True)
    print(f"{'MEAN':<28}{m['mae']:>8.3f}{m['r2']:>8.3f}{m['base_mae']:>10.3f}{m['base_r2']:>9.3f}")
    print(f"{'std across folds':<28}{folds['mae'].std():>8.3f}{folds['r2'].std():>8.3f}")
    return m


def make_plot(df, ridge, compounds, colors):
    """Before/after: Day 1's raw means next to the fuel-corrected Ridge curves."""
    fig, (ax_raw, ax_fit) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    grouped = df.groupby(["Compound", "TyreLife"])["Delta"].agg(["mean", "size"])
    grouped = grouped[grouped["size"] >= MIN_LAPS].reset_index()
    for compound, g in grouped.groupby("Compound"):
        g = g.sort_values("TyreLife")
        ax_raw.plot(g["TyreLife"], g["mean"], marker="o", ms=3,
                    color=colors.get(compound), label=compound)
    ax_raw.set_title("BEFORE: raw means (fuel burn not removed)")
    ax_raw.set_ylabel("Pace loss vs driver's best lap (s)")

    # Fitted curves at mean conditions, so the age term is what varies.
    mean_row = {"FuelPct": df["FuelPct"].mean(), "TrackTemp": df["TrackTemp"].mean()}
    for compound in compounds:
        ages = np.arange(1, int(df.loc[df["Compound"] == compound, "TyreLife"].quantile(0.98)) + 1)
        grid = pd.DataFrame({"TyreLife": ages, "Compound": compound, **mean_row})
        ax_fit.plot(ages, ridge.predict(ridge_design(grid, compounds)),
                    color=colors.get(compound), lw=2, label=compound)
    ax_fit.set_title("AFTER: Ridge fit, fuel + track temp held at mean")

    for ax in (ax_raw, ax_fit):
        ax.set_xlabel("Tyre age (laps)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Tyre degradation before and after fuel correction")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nsaved {OUT_PNG}")


def main():
    df = add_features(pd.read_csv(IN_CSV))
    assert not df.empty, f"{IN_CSV} loaded empty — run fetch_data.py first"

    compounds = sorted(df["Compound"].dropna().unique())
    print(f"{len(df)} laps, {df['Race'].nunique()} races, compounds: {', '.join(compounds)}")
    print("laps per compound:\n" + df["Compound"].value_counts().to_string())

    y = df["Delta"]
    X_ridge = ridge_design(df, compounds)
    X_xgb = pd.get_dummies(df[XGB_FEATURES + ["Compound"]], columns=["Compound"], dtype=float)

    # Self-check 1: nothing NaN reached either model.
    for name, X in (("ridge", X_ridge), ("xgb", X_xgb)):
        bad = X.columns[X.isna().any()].tolist()
        assert not bad, f"NaNs leaked into {name} features: {bad}"
    assert not y.isna().any(), "NaNs in target Delta"

    # Wet races are "effectively a separate regime rather than a covariate shift" (doc
    # Part 4), so score twice: everything pooled, and dry races only. Wet races stay in
    # the descriptive dataset and the chart either way — they just leave the CV rotation.
    per_race = df.groupby("Race").agg(rain=("Rainfall", "mean"), spread=("Delta", "std"),
                                      laps=("Delta", "size"))
    wet = per_race.index[per_race["rain"] > WET_FRACTION].tolist()
    print(f"\n--- per race (wet = rain on >{WET_FRACTION:.0%} of clean laps) ---")
    print(f"{'race':<28}{'laps':>7}{'rain frac':>11}{'Delta std':>11}{'':>6}")
    for race, r in per_race.iterrows():
        print(f"{race:<28}{int(r['laps']):>7}{r['rain']:>11.3f}{r['spread']:>11.3f}"
              f"{'  WET' if race in wet else '':>6}")

    views = {"pooled": pd.Series(True, index=df.index), "dry only": ~df["Race"].isin(wet)}
    means = {}
    for view, mask in views.items():
        for name, X, make in (("Ridge", X_ridge, make_ridge), ("XGBoost", X_xgb, make_xgb)):
            folds = cv_scores(X[mask.values], y[mask.values], df.loc[mask.values, "Race"], make)
            means[(name, view)] = print_folds(f"{name}, {view}", folds)

    # Coefficients come from a full-data refit — the CV above is what measures accuracy.
    ridge = make_ridge().fit(X_ridge, y)
    coef = pd.Series(ridge.coef_, index=X_ridge.columns)

    # age and age^2/100 are strongly collinear, so the split between them is unstable and
    # the linear coefficient alone does not say whether the curve rises. The average
    # gradient across the compound's observed age range does, so report both.
    net = {}
    for c in compounds:
        hi = df.loc[df["Compound"] == c, "TyreLife"].quantile(0.98)
        b1, b2 = coef[f"age[{c}]"], coef[f"age2[{c}]"]
        net[c] = ((b1 * hi + b2 * hi**2 / 100) - (b1 + b2 / 100)) / (hi - 1)

    print("\n--- Ridge coefficients (full-data refit, quotable) ---")
    print(f"{'compound':<16}{'intercept':>11}{'age s/lap':>12}{'age^2/100':>12}{'net s/lap':>12}")
    for c in compounds:
        print(f"{c:<16}{coef[f'int[{c}]']:>11.3f}{coef[f'age[{c}]']:>12.4f}"
              f"{coef[f'age2[{c}]']:>12.4f}{net[c]:>12.4f}")
    print("net s/lap = average gradient of the fitted curve over that compound's age range")
    print(f"\nfuel effect  {coef['FuelPct']:+.3f} s across a full tank (FuelPct 1 -> 0)")
    print(f"track temp   {coef['TrackTemp']:+.4f} s per degC")

    make_plot(df, ridge, compounds, dict(zip(compounds, plt.cm.tab10.colors)))

    # --- pass/fail summary ---
    print("\n" + "=" * 62)
    checks = [
        (f"rows loaded ({len(df)})", len(df) > 0),
        ("no NaNs in features/target", True),  # asserted above, would not reach here
        *[(f"{name} {view}: MAE {m['mae']:.3f} < baseline {m['base_mae']:.3f}",
           m["mae"] < m["base_mae"]) for (name, view), m in means.items()],
        ("every fitted curve rises with tyre age (degradation, not gain)",
         all(v > 0 for v in net.values())),
    ]
    for label, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    print("=" * 62)
    print(f"{sum(ok for _, ok in checks)}/{len(checks)} checks passed"
          f"{'' if all(ok for _, ok in checks) else '  <-- see FAIL lines above'}")
    print(f"caveat: {df['Race'].nunique()} races = that many CV groups, so per-fold "
          f"numbers are noisy by construction.")


if __name__ == "__main__":
    main()
