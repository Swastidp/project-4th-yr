"""
Tier 1 chart: mean raw pace loss vs tyre age,
one line per compound.

Target:
    Lap delta vs the driver's own best lap in that race.

This chart intentionally shows the RAW relationship before
fuel correction. It is used as the "before" visual in the
Stage-1 degradation analysis.

Run:
    uv run plot_degradation.py
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).parent / "data"

# Ignore tyre-age points supported by too few laps.
MIN_LAPS = 20


def main():
    input_file = DATA_DIR / "laps_clean.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"{input_file} not found. Run fetch_data.py first.")

    df = pd.read_csv(input_file)

    if df.empty:
        raise RuntimeError(f"{input_file} is empty.")

    # -----------------------------------------------------
    # Pace loss
    # -----------------------------------------------------
    # Lap time minus the driver's own best lap
    # in that race.

    best = df.groupby(["Year", "Event", "Driver"])["LapTimeSeconds"].transform("min")

    df["PaceLoss"] = df["LapTimeSeconds"] - best

    # -----------------------------------------------------
    # Mean pace loss by compound and tyre age
    # -----------------------------------------------------

    grouped = (
        df.groupby(["Compound", "TyreLife"])["PaceLoss"]
        .agg(
            mean="mean",
            size="size",
        )
        .reset_index()
    )

    grouped = grouped[grouped["size"] >= MIN_LAPS].copy()

    if grouped.empty:
        raise RuntimeError(
            f"No tyre-age groups remain after the MIN_LAPS={MIN_LAPS} filter."
        )

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for compound, group in grouped.groupby("Compound"):
        group = group.sort_values("TyreLife")

        ax.plot(
            group["TyreLife"],
            group["mean"],
            marker="o",
            ms=3,
            label=(f"{compound} (n={int(group['size'].sum())})"),
        )

    ax.set_xlabel("Tyre age (laps)")

    ax.set_ylabel("Mean pace loss vs driver's best lap (s)")

    ax.set_title("Raw tyre pace loss vs tyre age by compound")

    ax.legend()

    ax.grid(alpha=0.3)

    fig.tight_layout()

    output_file = DATA_DIR / "degradation_curve.png"

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print(f"saved {output_file}\n")

    print(f"{'compound':<14}{'age range':>12}{'early':>9}{'late':>9}{'trend':>12}")

    for compound, group in grouped.groupby("Compound"):
        group = group.sort_values("TyreLife")

        third = max(
            len(group) // 3,
            1,
        )

        early = group["mean"].head(third).mean()

        late = group["mean"].tail(third).mean()

        if late > early:
            trend = "increasing"
        elif late < early:
            trend = "decreasing"
        else:
            trend = "flat"

        age_range = f"{int(group['TyreLife'].min())}-{int(group['TyreLife'].max())}"

        print(f"{compound:<14}{age_range:>12}{early:>9.2f}{late:>9.2f}{trend:>12}")

    print(
        "\nNote: this is the uncorrected raw relationship. "
        "Fuel burn can mask tyre degradation, so the "
        "fuel-corrected Ridge curves from train_pace_loss.py "
        "are used for the corrected degradation analysis."
    )


if __name__ == "__main__":
    main()
