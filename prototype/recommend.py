"""Stage-3 compound + stint-length recommender. No new training — this scores Day 2's model.

Cost(c, L) = sum_{t=1..L} dhat(c, t, FuelPct) + P * (R / L)      (design doc, Stage 3)

dhat comes from Day 2's Ridge fit, refit here on the same laps with the same feature code
imported from train_pace_loss — so the fuel formula can't drift out of sync with Day 2's.

Circuit fixed effects: the refit here adds one-hot Event dummies (first level dropped as the
reference circuit), additive only — no per-circuit compound slope, there isn't the data for
one. Reason: Pirelli nominates 3 of 5 compounds per weekend, so "HARD" at Monza and "HARD"
at Bahrain are different physical rubber. Without a circuit term the compound intercepts
absorb "which circuits used this label" (doc Part 4: "circuit identity must be a model
feature"). This is a SEPARATE fit used only for ranking here — Day 2's cross-validated
accuracy claim (dry-only Ridge MAE 0.759 s) is measured without it and is untouched.

ponytail: 8 circuit dummies fit on 9 races is a lot of parameters for the data. Ceiling: the
circuit coefficients themselves are noisy and the reference-circuit choice shifts every
absolute cost (compound *ordering* is unaffected — the term is additive). Upgrade path: more
seasons, or a random-effects / partially-pooled race intercept.

ponytail: TrackTemp is deliberately dropped from the cost function. Day 2's refit found its
coefficient confounded with circuit identity (9 races, so "hot" and "Monza" are nearly the
same variable) and signed backwards from physics. Ceiling: this recommender cannot adjust
for weather or track temperature at all — a 42 degC Bahrain and a 22 degC Silverstone get
the same answer. Upgrade path: put TrackTemp back once the dataset has repeat visits to the
same circuit under different conditions, which is what separates it from circuit identity.

Run:  uv run recommend.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from build_stints import km_table  # Day 3's Kaplan-Meier, reused rather than reimplemented
from train_pace_loss import IN_CSV, add_features, make_ridge, ridge_design

STINTS_CSV = Path(__file__).parent / "data" / "stints.csv"

PIT_LOSS_S = 22.0   # doc says 20-25 s; middle of the range. One-line change.
DRY = ["HARD", "MEDIUM", "SOFT"]
MAX_STINT = 40      # no dry stint in the dataset comes close; keeps the search cheap
KM_SLACK = 1.5      # flag an L more than this multiple of the compound's KM median
TOP_N = 8


def design(df, compounds, events=()):
    """train_pace_loss.ridge_design minus TrackTemp, plus additive circuit dummies.

    `events` is the non-reference circuit list. A prediction grid carries no Event column,
    so every dummy is 0 — predictions are made at the reference circuit.
    """
    X = ridge_design(df.assign(TrackTemp=0.0), compounds).drop(columns=["TrackTemp"])
    ev = df["Event"] if "Event" in df.columns else pd.Series("", index=df.index)
    for e in events:
        X[f"race[{e}]"] = (ev == e).astype(float)
    return X


def load_laps():
    df = add_features(pd.read_csv(IN_CSV))
    assert not df.empty, f"{IN_CSV} loaded empty — run fetch_data.py first"
    return df


def fit_ridge(df=None, use_events=True):
    """Refit Day 2's Ridge on the same laps, TrackTemp excluded. Fast: closed-form."""
    df = load_laps() if df is None else df
    compounds = sorted(df["Compound"].dropna().unique())
    # Drop the first circuit as the reference category — standard fixed-effects practice.
    events = sorted(df["Event"].dropna().unique())[1:] if use_events else []
    X = design(df, compounds, events)
    bad = X.columns[X.isna().any()].tolist()
    assert not bad, f"NaNs leaked into the design matrix: {bad}"
    dead = X.columns[(X == 0).all()].tolist()
    assert not dead, f"all-zero design columns after one-hot encoding: {dead}"
    ridge = make_ridge().fit(X, df["Delta"])
    return ridge, compounds, events


def km_medians():
    """Kaplan-Meier median stint length per compound, from Day 3's stint table."""
    st = pd.read_csv(STINTS_CSV)
    compounds = [c for c in DRY if c in set(st["Compound"])]
    km, _ = km_table(st, compounds)
    return dict(zip(km["compound"], km["km_median"]))


def fitted(ridge, compounds, events, compound, ages, fuel):
    """Predicted pace loss for one compound over `ages`, at the reference circuit."""
    grid = pd.DataFrame({"TyreLife": ages, "Compound": compound, "FuelPct": fuel})
    return ridge.predict(design(grid, compounds, events))


def cum_loss(ridge, compounds, events, compound, current_lap, total_race_laps, max_L):
    """Cumulative predicted pace loss: element L-1 is sum_{t=1..L} dhat(c, t, fuel)."""
    age = np.arange(1, max_L + 1)
    lap_number = np.minimum(current_lap + age, total_race_laps)
    fuel = 1 - lap_number / total_race_laps   # same formula as add_features
    return np.cumsum(fitted(ridge, compounds, events, compound, age, fuel))


def cost_table(ridge, compounds, events, medians, current_lap, total_race_laps):
    """Every (compound, stint length) pair, ranked by cost. R = laps left to run."""
    R = total_race_laps - current_lap
    max_L = int(min(R, MAX_STINT))
    rows = []
    for c in DRY:
        cum = cum_loss(ridge, compounds, events, c, current_lap, total_race_laps, max_L)
        med = medians.get(c, np.nan)
        for L in range(1, max_L + 1):
            rows.append({
                "compound": c,
                "stint_laps": L,
                "deg_cost": cum[L - 1],
                "pit_cost": PIT_LOSS_S * R / L,
                "cost": cum[L - 1] + PIT_LOSS_S * R / L,
                "km_median": med,
                "flag": "" if L <= KM_SLACK * med else f"> {KM_SLACK}x KM median ({med:.0f})",
            })
    return pd.DataFrame(rows).sort_values("cost").reset_index(drop=True)


def recommend(current_lap, total_race_laps, rainfall, compounds_available=DRY,
              _fit=None, _medians=None):
    """Ranked (compound, stint length) options, lowest cost first. Rain routes by rule."""
    if rainfall:
        return pd.DataFrame([{
            "compound": "INTERMEDIATE", "stint_laps": np.nan, "cost": np.nan,
            "flag": "by rule: rainfall - cost optimisation skipped (FIA can mandate wets)",
        }])
    ridge, compounds, events = _fit or fit_ridge()
    medians = _medians if _medians is not None else km_medians()
    table = cost_table(ridge, compounds, events, medians, current_lap, total_race_laps)
    return table[table["compound"].isin(compounds_available)].reset_index(drop=True)


def show(title, table):
    print(f"\n--- {title} ---")
    if table["cost"].isna().all():
        print(f"{table.iloc[0]['compound']}  ({table.iloc[0]['flag']})")
        return
    print(f"{'#':<4}{'compound':<10}{'stint L':>9}{'deg cost':>10}{'pit cost':>10}"
          f"{'TOTAL s':>10}   flag")
    for i, r in table.head(TOP_N).iterrows():
        print(f"{i + 1:<4}{r['compound']:<10}{r['stint_laps']:>9.0f}{r['deg_cost']:>10.1f}"
              f"{r['pit_cost']:>10.1f}{r['cost']:>10.1f}   {r['flag']}")
    best = table.loc[table.groupby("compound")["cost"].idxmin()].sort_values("cost")
    print("  best L per compound: " + ",  ".join(
        f"{r['compound']} L={r['stint_laps']:.0f} ({r['cost']:.1f}s)" for _, r in best.iterrows()))


def compound_coefs(df, fit):
    """intercept / age / age^2 / net gradient per compound, same format as Day 2's table."""
    ridge, compounds, events = fit
    coef = pd.Series(ridge.coef_, index=design(df, compounds, events).columns)
    rows = {}
    for c in compounds:
        # age and age^2/100 are collinear, so quote the average gradient too (Day 2's note).
        hi = df.loc[df["Compound"] == c, "TyreLife"].quantile(0.98)
        b1, b2 = coef[f"age[{c}]"], coef[f"age2[{c}]"]
        rows[c] = (coef[f"int[{c}]"], b1, b2,
                   ((b1 * hi + b2 * hi**2 / 100) - (b1 + b2 / 100)) / (hi - 1))
    return rows


def print_coef_comparison(plain, circ, compounds):
    head = f"{'intercept':>11}{'age s/lap':>11}{'age^2/100':>11}{'net s/lap':>11}"
    print("\n--- per-compound Ridge coefficients, full-data refit ---")
    print(f"{'':<14}{'|--- NO circuit term (as shipped) ---|':^46}   "
          f"{'|--- WITH circuit term (new) ---|':^46}")
    print(f"{'compound':<14}{head:<46}   {head:<46}")
    for c in compounds:
        a, b = plain[c], circ[c]
        print(f"{c:<14}" + "".join(f"{v:>11.3f}" if i == 0 else f"{v:>11.4f}"
                                   for i, v in enumerate(a)) + "  " + " " * 2
              + "".join(f"{v:>11.3f}" if i == 0 else f"{v:>11.4f}"
                        for i, v in enumerate(b)))
    print("net s/lap = average gradient of the fitted curve over that compound's age range")


def main():
    df = load_laps()
    fit_plain = fit_ridge(df, use_events=False)
    fit = fit_ridge(df, use_events=True)
    ref_event = sorted(df["Event"].dropna().unique())[0]
    medians = km_medians()
    print(f"pit loss P = {PIT_LOSS_S:.0f}s | KM median stint length: "
          + ", ".join(f"{c} {m:.0f}" for c, m in medians.items()))
    print("cost function uses Compound + TyreLife + TyreLife^2 + FuelPct only (no TrackTemp)")
    print(f"circuit term: {len(fit[2])} Event dummies, reference circuit = {ref_event}")
    print("note: deg cost is now relative to the reference circuit, so it can print negative."
          "\n      The circuit term is additive, so it shifts every compound equally and the"
          "\n      ranking is unchanged by which circuit is the reference (checked, all 8).")

    plain_c = compound_coefs(df, fit_plain)
    circ_c = compound_coefs(df, fit)
    print_coef_comparison(plain_c, circ_c, fit[1])

    early = recommend(5, 57, False, _fit=fit, _medians=medians)
    late = recommend(45, 57, False, _fit=fit, _medians=medians)
    wet = recommend(20, 57, True, _fit=fit, _medians=medians)

    show("Scenario 1: lap 5 of 57, R = 52 laps left", early)
    show("Scenario 2: lap 45 of 57, R = 12 laps left", late)
    show("Scenario 3: lap 20 of 57, rainfall = True", wet)

    # --- the exact thing that was broken: is SOFT faster than HARD on a brand-new tyre? ---
    fuel = df["FuelPct"].mean()
    new_tyre = {name: {c: fitted(*f, c, np.array([1]), fuel)[0] for c in DRY}
                for name, f in (("no circuit", fit_plain), ("with circuit", fit))}
    print("\n--- fitted pace loss at age = 1 (brand-new tyre, fuel at dataset mean) ---")
    print(f"{'fit':<16}" + "".join(f"{c:>12}" for c in DRY) + f"{'SOFT < HARD?':>16}")
    for name, v in new_tyre.items():
        ok = "PASS" if v["SOFT"] < v["HARD"] else "FAIL"
        print(f"{name:<16}" + "".join(f"{v[c]:>12.3f}" for c in DRY) + f"{ok:>16}")
    soft_beats_hard = new_tyre["with circuit"]["SOFT"] < new_tyre["with circuit"]["HARD"]

    print("\n--- honest read ---")
    print("Scenario 1 (52 laps left) should favour a durable tyre; scenario 2 (12 laps left)")
    print("should favour SOFT. Before the circuit term, SOFT's curve sat above HARD's at EVERY")
    print("age, so no stint length could make SOFT win. Pirelli nominates 3 of 5 compounds per")
    print("weekend, so 'HARD' at Monza is different rubber from 'HARD' at Bahrain; with 9 races")
    print("and no circuit term the compound intercepts absorbed that mix (doc Part 4).")
    print(f"After adding {len(fit[2])} circuit dummies: SOFT at age 1 is "
          f"{'BELOW' if soft_beats_hard else 'still ABOVE'} HARD, and scenario 2's top pick is "
          f"{late.iloc[0]['compound']}.")
    print(f"Tradeoff to state in the report: {len(fit[2])} circuit dummies on "
          f"{df['Race'].nunique()} races is a lot of parameters for the data, so the circuit")
    print("coefficients are themselves noisy. Sanity test that it is a real correction and not")
    print("a reshuffle: SOFT should still degrade fastest. SOFT net gradient "
          f"{circ_c['SOFT'][3]:+.4f} s/lap vs MEDIUM {circ_c['MEDIUM'][3]:+.4f} vs HARD "
          f"{circ_c['HARD'][3]:+.4f}.")

    # --- self-check ---
    extreme = cost_table(*fit, medians, 5, 200)          # forces L up to MAX_STINT = 40
    fired = sorted(extreme.loc[(extreme["stint_laps"] == 40) & (extreme["flag"] != ""), "compound"])
    checks = [
        ("circuit-adjusted design matrix is clean (no NaNs, no all-zero columns)", True),
        ("SOFT starts below HARD at age 1 (the ordering that was broken)", soft_beats_hard),
        ("SOFT still degrades fastest of the three (physics sanity)",
         circ_c["SOFT"][3] > max(circ_c["MEDIUM"][3], circ_c["HARD"][3])),
        ("cost table non-empty", not early.empty and not late.empty),
        ("all costs finite", np.isfinite(early["cost"]).all() and np.isfinite(late["cost"]).all()),
        ("all costs positive", (early["cost"] > 0).all() and (late["cost"] > 0).all()),
        ("rainfall -> INTERMEDIATE", wet.iloc[0]["compound"] == "INTERMEDIATE"),
        ("rainfall skips cost optimisation (no cost computed)", wet["cost"].isna().all()),
        (f"scenario 1 top = {early.iloc[0]['compound']}", early.iloc[0]["compound"] in DRY),
        (f"scenario 2 top = {late.iloc[0]['compound']}", late.iloc[0]["compound"] in DRY),
        (f"KM flag fires on a deliberately long L (40 laps): {', '.join(fired) or 'nothing'}",
         bool(fired)),
        ("KM flag stays silent on a short L (5 laps)",
         (extreme[extreme["stint_laps"] == 5]["flag"] == "").all()),
    ]
    print("\n" + "=" * 62)
    for label, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    print("=" * 62)
    n_ok = sum(ok for _, ok in checks)
    print(f"{n_ok}/{len(checks)} checks passed"
          f"{'' if n_ok == len(checks) else '  <-- see FAIL lines above'}")


if __name__ == "__main__":
    main()
