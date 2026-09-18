from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent / "data"
POINTS_TO_KEEP = 6


def point_distance(point, start, end):
    """
    Calculate the vertical distance of a point
    from the straight line between start and end.
    """

    x, y = point
    x1, y1 = start
    x2, y2 = end

    if x2 == x1:
        return abs(y - y1)

    expected_y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)

    return abs(y - expected_y)


def simplify_curve(df, target_points=POINTS_TO_KEEP):
    """
    Keep the most important points while preserving
    the overall shape of the curve.
    """

    if df.empty:
        return df.copy()

    if target_points < 2:
        raise ValueError("target_points must be at least 2.")

    if len(df) <= target_points:
        return df.copy()

    points = df[["TyreLife", "PaceLoss"]].to_numpy(dtype=float)

    # Always keep first and last points.
    selected = {
        0,
        len(points) - 1,
    }

    while len(selected) < target_points:
        selected_sorted = sorted(selected)

        best_index = None
        best_distance = -1.0

        # Examine every section between
        # already-selected points.
        for i in range(len(selected_sorted) - 1):
            start_index = selected_sorted[i]
            end_index = selected_sorted[i + 1]

            if end_index - start_index <= 1:
                continue

            start = points[start_index]
            end = points[end_index]

            for j in range(
                start_index + 1,
                end_index,
            ):
                distance = point_distance(
                    points[j],
                    start,
                    end,
                )

                if distance > best_distance:
                    best_distance = distance
                    best_index = j

        if best_index is None:
            break

        selected.add(best_index)

    selected = sorted(selected)

    return df.iloc[selected].copy()


def process_file(input_file):
    """
    Reduce one degradation-curve CSV and save the
    simplified version as main_<filename>.
    """

    if not input_file.exists():
        print(f"SKIPPED: {input_file} not found")
        return

    df = pd.read_csv(input_file)

    required = [
        "TyreLife",
        "PaceLoss",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"{input_file.name}: missing required columns: {missing}")

    if df.empty:
        print(f"SKIPPED: {input_file.name} is empty")
        return

    # Ensure the curve is ordered by tyre age.
    df["TyreLife"] = pd.to_numeric(
        df["TyreLife"],
        errors="coerce",
    )

    df["PaceLoss"] = pd.to_numeric(
        df["PaceLoss"],
        errors="coerce",
    )

    df = (
        df.dropna(
            subset=[
                "TyreLife",
                "PaceLoss",
            ]
        )
        .sort_values("TyreLife")
        .reset_index(drop=True)
    )

    if df.empty:
        print(f"SKIPPED: {input_file.name} has no valid TyreLife/PaceLoss rows")
        return

    simplified = simplify_curve(
        df,
        POINTS_TO_KEEP,
    )

    output_file = DATA_DIR / f"main_{input_file.name}"

    simplified.to_csv(
        output_file,
        index=False,
    )

    print(f"{input_file.name}: {len(df)} points -> {len(simplified)} points")

    print(f"saved: {output_file}")


def main():

    files = [
        "before_hard.csv",
        "before_intermediate.csv",
        "before_medium.csv",
        "before_soft.csv",
        "after_hard.csv",
        "after_intermediate.csv",
        "after_medium.csv",
        "after_soft.csv",
    ]

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename in files:
        process_file(DATA_DIR / filename)


if __name__ == "__main__":
    main()
