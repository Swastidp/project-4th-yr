"""Stage-2 stint-life model: how long a tyre stint survives, as survival analysis.

A stint ends because the tyre degraded, the race finished, or the driver retired. Only the
first is an "event"; the other two are right-censored. Treating all three the same is why a
plain stint-length regressor scores R^2 ~ 0 (design doc, Stage 2).

Deliberately does NOT read data/laps_clean.csv: its three quicklap filters drop in/out laps
and safety-car laps, which would shorten the observed stint lengths. Survival needs every lap
the tyre was actually on the car, so this reloads RAW session.laps from the FastF1 cache.

Run:  uv run build_stints.py
"""

import time
import warnings
from pathlib import Path

import fastf1
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: E402
from lifelines.exceptions import ConvergenceError  # noqa: E402

from fetch_data import CACHE_DIR, RACES, WEATHER_COLS  # noqa: E402  same 9 races, same cache

DATA_DIR = Path(__file__).parent / "data"
OUT_CSV = DATA_DIR / "stints.csv"
OUT_PNG = DATA_DIR / "survival_curves.png"

DRY = ["HARD", "MEDIUM", "SOFT"]  # the compounds that always get their own KM curve
MIN_STINTS_FOR_COX = 20  # below this a compound dummy is fitted on noise; doc had 44 WET stints
COX_COVARIATES = ["is_soft", "is_medium", "is_intermediate", "start_lap", "total_race_laps",
                  "Humidity", "TrackTemp", "starting_tyre_age"]


def stints_for_race(year, event):
    """One row per (Driver, Stint) from RAW laps — no quicklap filters. Returns (df, resolved)."""
    session = fastf1.get_session(year, event, "R")
    session.load(telemetry=False, messages=False)
    laps = session.laps.copy()

    # session.total_laps is the scheduled distance; max LapNumber is what was actually run
    # (they differ if the race was red-flagged short). Take the max of what we can see.
    total_race_laps = int(max(laps["LapNumber"].max(), getattr(session, "total_laps", 0) or 0))

    g = laps.sort_values("LapNumber").groupby(["Driver", "Stint"], dropna=True)
    st = g.agg(
        stint_length=("LapNumber", "size"),
        start_lap=("LapNumber", "min"),
        end_lap=("LapNumber", "max"),
        starting_tyre_age=("TyreLife", "first"),
        start_time=("LapStartTime", "first"),
        Compound=("Compound", "first"),
        n_compounds=("Compound", "nunique"),
        pit_in=("PitInTime", lambda s: s.notna().sum()),
    ).reset_index()

    # Weather at the stint's first lap. Lap 1 has a NaT LapStartTime, so anchor it at t=0 and
    # let "nearest" pick the first weather sample — which is the right one for a lap-1 stint.
    weather = session.weather_data[["Time"] + WEATHER_COLS].sort_values("Time")
    st["start_time"] = st["start_time"].fillna(pd.Timedelta(0))
    st = pd.merge_asof(st.sort_values("start_time"), weather,
                       left_on="start_time", right_on="Time", direction="nearest")

    st["total_race_laps"] = total_race_laps
    st.insert(0, "Event", session.event["EventName"])
    st.insert(0, "Year", year)
    return st, session.event["EventName"]


def add_censoring(st):
    """event=1 if a later stint exists for that driver (so this one ended in a pit stop)."""
    key = ["Year", "Event", "Driver"]
    st["event"] = (st["Stint"] < st.groupby(key)["Stint"].transform("max")).astype(int)
    # For the censored (last) stints: did the driver see the flag, or did they stop early?
    st["finished_race"] = st["end_lap"] >= st["total_race_laps"] - 1  # -1: lapped cars
    return st


def km_table(st, compounds):
    """Kaplan-Meier median vs the plain uncensored median, per compound."""
    rows = []
    for c in compounds:
        g = st[st["Compound"] == c]
        kmf = KaplanMeierFitter().fit(g["stint_length"], g["event"], label=c)
        rows.append({
            "compound": c, "stints": len(g), "events": int(g["event"].sum()),
            "censored_pct": 100 * (1 - g["event"].mean()),
            "raw_mean": g["stint_length"].mean(), "raw_median": g["stint_length"].median(),
            "km_median": kmf.median_survival_time_,
        })
    return pd.DataFrame(rows), {c: KaplanMeierFitter().fit(
        st.loc[st["Compound"] == c, "stint_length"],
        st.loc[st["Compound"] == c, "event"], label=c) for c in compounds}


def make_plot(fitters):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c, kmf in fitters.items():
        kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_xlabel("Stint length (laps)")
    ax.set_ylabel("P(stint still running)")
    ax.set_title("Kaplan-Meier stint survival by compound")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nsaved {OUT_PNG}")


def main():
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    # Every race was already downloaded on day 1/2. Offline mode turns a silent re-download
    # into a loud error, which is what we want — see the except below.
    fastf1.Cache.offline_mode(True)

    frames = []
    t0 = time.time()
    for year, event in RACES:
        try:
            st, resolved = stints_for_race(year, event)
        except Exception as exc:  # noqa: BLE001 — any cache miss must stop the run, not retry online
            raise SystemExit(
                f"\nCACHE MISS on {year} {event}: {type(exc).__name__}: {exc}\n"
                f"Offline mode is on, so nothing was re-downloaded. Re-run fetch_data.py to "
                f"populate {CACHE_DIR} for this race, then re-run this script."
            ) from exc
        print(f"{year} {resolved:<34} {len(st):>4} stints, {st['Driver'].nunique():>2} drivers")
        frames.append(st)
    print(f"loaded {len(RACES)} sessions from cache in {time.time() - t0:.1f}s (no network)")

    st = add_censoring(pd.concat(frames, ignore_index=True))

    # A stint whose compound changes mid-stint means the grouping key is wrong, not that the
    # tyre changed. Loud, because every downstream number depends on it.
    bad = st[st["n_compounds"] > 1]
    assert bad.empty, f"{len(bad)} stints span >1 compound:\n{bad.head().to_string()}"

    st = st.dropna(subset=["Compound"])
    st = st[~st["Compound"].isin(["UNKNOWN", "TEST_UNKNOWN"])]
    st.to_csv(OUT_CSV, index=False)

    print(f"\n{len(st)} stints, {st.groupby(['Year', 'Event']).ngroups} races, "
          f"{st.groupby(['Year', 'Event', 'Driver']).ngroups} driver-races -> {OUT_CSV}")
    cens = st[st["event"] == 0]
    print(f"censored: {len(cens)} of {len(st)} ({1 - st['event'].mean():.1%}) — doc reference ~21%")
    print(f"  of those, {cens['finished_race'].sum()} reached the flag, "
          f"{(~cens['finished_race']).sum()} stopped early (retirement/red flag)")

    # --- data-quality flags. Not filtered — these are real laps — but they must be visible. ---
    short = st[st["stint_length"] <= 2]
    print(f"\ndata quality:")
    print(f"  {len(short)} stints of <=2 laps, {(short['Event'] == 'Japanese Grand Prix').sum()} "
          f"of them at Suzuka — the 2024 red flag on lap 1 let everyone change tyres for free")
    print(f"  {len(cens[cens['pit_in'] > 0])} censored stints show a pit entry on their last lap "
          f"(pitted, then retired) — kept as censored, see pit_in in {OUT_CSV.name} to revisit")
    for c in sorted(set(st["Compound"]) - set(DRY)):
        by_race = st.loc[st["Compound"] == c].groupby(["Year", "Event"]).size().sort_values()
        print(f"  {c}: {by_race.sum()} stints from {len(by_race)} race(s), "
              f"{by_race.iloc[-1] / by_race.sum():.0%} from one race — not a compound effect")
    print(f"  starting_tyre_age is 1-based: {(st['starting_tyre_age'] == 1).sum()} stints start "
          f"fresh (=1), {(st['starting_tyre_age'] > 1).sum()} on a used set (2-"
          f"{int(st['starting_tyre_age'].max())})")

    # --- Kaplan-Meier ---
    compounds = [c for c in DRY if c in set(st["Compound"])]
    compounds += sorted(set(st["Compound"]) - set(DRY))
    km, fitters = km_table(st, compounds)
    print("\n--- Kaplan-Meier vs raw stint length (laps) ---")
    print(f"{'compound':<14}{'stints':>7}{'events':>8}{'cens %':>8}{'raw mean':>10}"
          f"{'raw med':>9}{'KM med':>8}{'KM-raw':>8}")
    for _, r in km.iterrows():
        gap = r["km_median"] - r["raw_median"]
        print(f"{r['compound']:<14}{r['stints']:>7.0f}{r['events']:>8.0f}{r['censored_pct']:>7.0f}%"
              f"{r['raw_mean']:>10.1f}{r['raw_median']:>9.1f}{r['km_median']:>8.1f}{gap:>+8.1f}")
    print("KM median should sit at or above the raw median — censoring drags the raw one down")

    # --- Cox PH ---
    counts = st["Compound"].value_counts()
    keep = [c for c in compounds if counts[c] >= MIN_STINTS_FOR_COX or c == "HARD"]
    dropped = [f"{c} (n={counts[c]})" for c in compounds if c not in keep]
    cox_df = st[st["Compound"].isin(keep)].copy()
    for c in ["SOFT", "MEDIUM", "INTERMEDIATE"]:  # HARD is the baseline
        cox_df[f"is_{c.lower()}"] = (cox_df["Compound"] == c).astype(float)
    print(f"\n--- Cox PH (baseline HARD) ---")
    print(f"excluded from the fit: {', '.join(dropped) if dropped else 'nothing'} "
          f"(threshold: <{MIN_STINTS_FOR_COX} stints)")

    cols = [c for c in COX_COVARIATES if cox_df[c].nunique() > 1]
    cox_df = cox_df[cols + ["stint_length", "event"]].dropna()
    cph = CoxPHFitter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            cph.fit(cox_df, duration_col="stint_length", event_col="event")
        except ConvergenceError as exc:
            raise SystemExit(f"Cox model did not converge: {exc}") from exc
    for w in caught:
        print(f"  lifelines warning: {str(w.message).splitlines()[0][:110]}")

    summary = cph.summary
    print(f"\n{'covariate':<22}{'coef':>9}{'hazard ratio':>14}{'p':>10}{'':>4}")
    for name, r in summary.iterrows():
        star = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        print(f"{name:<22}{r['coef']:>9.3f}{r['exp(coef)']:>14.2f}{r['p']:>10.3f}{star:>4}")
    print(f"\nC-index {cph.concordance_index_:.3f}  (0.5 = random; doc reference 0.672)")
    print(f"fitted on {len(cox_df)} stints, {int(cox_df['event'].sum())} events")

    make_plot(fitters)

    # --- pass/fail summary ---
    print("\n" + "=" * 62)
    checks = [
        (f"stint table non-empty ({len(st)} stints)", not st.empty),
        (f"censoring detected, not degenerate ({st['event'].mean():.1%} events)",
         0 < st["event"].mean() < 1),
        ("Cox model converged", hasattr(cph, "params_") and cph.params_.notna().all()),
        (f"C-index in (0, 1): {cph.concordance_index_:.3f}", 0 < cph.concordance_index_ < 1),
        ("softer compound = higher hazard than HARD (physically sensible)",
         summary.loc["is_soft", "exp(coef)"] > summary.loc["is_medium", "exp(coef)"] > 1),
    ]
    for label, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    print("=" * 62)
    n_ok = sum(ok for _, ok in checks)
    print(f"{n_ok}/{len(checks)} checks passed"
          f"{'' if n_ok == len(checks) else '  <-- see FAIL lines above'}")
    print(f"caveat: {st.groupby(['Year', 'Event']).ngroups} races only — the doc's numbers come "
          f"from 2,289 stints, so treat every figure above as small-sample.")


if __name__ == "__main__":
    main()
