"""Tier 1 chart: mean pace loss vs tyre age, one line per compound.

Target is lap delta vs the driver's own best lap in that race — the one target the
design doc verified works (R2 0.348); raw lap time does not.

Run:  uv run plot_degradation.py
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
MIN_LAPS = 20  # drop tyre-age points backed by too few laps to mean anything


def main():
    df = pd.read_csv(DATA_DIR / "laps_clean.csv")

    # Pace loss = lap time minus that driver's best lap in that race.
    best = df.groupby(["Year", "Event", "Driver"])["LapTimeSeconds"].transform("min")
    df["PaceLoss"] = df["LapTimeSeconds"] - best

    grouped = df.groupby(["Compound", "TyreLife"])["PaceLoss"].agg(["mean", "size"])
    grouped = grouped[grouped["size"] >= MIN_LAPS].reset_index()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for compound, g in grouped.groupby("Compound"):
        g = g.sort_values("TyreLife")
        ax.plot(g["TyreLife"], g["mean"], marker="o", ms=3, label=f"{compound} (n={int(g['size'].sum())})")

    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Mean pace loss vs driver's best lap (s)")
    ax.set_title("Tyre degradation: pace loss vs tyre age by compound")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = DATA_DIR / "degradation_curve.png"
    fig.savefig(out, dpi=150)

    # Self-check: is the shape physically sensible? first vs last third of each curve.
    # ponytail: no fuel correction yet, so expect HARD/MEDIUM to read "DOWN" — the doc
    # measures fuel burn at ~-0.069 s/lap, which swamps real degradation. Fix by adding
    # FuelPct = 1 - LapNumber/TotalLaps as a feature in the Stage-1 model (Week 2 work).
    print(f"saved {out}\n")
    print(f"{'compound':<14}{'age range':>12}{'early':>9}{'late':>9}{'trend':>10}")
    for compound, g in grouped.groupby("Compound"):
        g = g.sort_values("TyreLife")
        third = max(len(g) // 3, 1)
        early, late = g["mean"].head(third).mean(), g["mean"].tail(third).mean()
        trend = "up (expected)" if late > early else "DOWN (check)"
        rng = f"{int(g['TyreLife'].min())}-{int(g['TyreLife'].max())}"
        print(f"{compound:<14}{rng:>12}{early:>9.2f}{late:>9.2f}{trend:>16}")


if __name__ == "__main__":
    main()
