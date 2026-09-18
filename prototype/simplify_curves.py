import pandas as pd
from pathlib import Path


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
    if len(df) <= target_points:
        return df.copy()

    points = df[["TyreLife", "PaceLoss"]].to_numpy()

    # Always keep first and last points.
    selected = {0, len(points) - 1}

    while len(selected) < target_points:
        selected_sorted = sorted(selected)

        best_index = None
        best_distance = -1

        # Check every section between already-selected points.
        for i in range(len(selected_sorted) - 1):
            start_index = selected_sorted[i]
            end_index = selected_sorted[i + 1]

            if end_index - start_index <= 1:
                continue

            start = points[start_index]
            end = points[end_index]

            for j in range(start_index + 1, end_index):
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
    df = pd.read_csv(input_file)

    # Make sure data is sorted by tyre age.
    df = df.sort_values("TyreLife").reset_index(drop=True)

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

    for filename in files:
        input_file = DATA_DIR / filename

        if not input_file.exists():
            print(f"SKIPPED: {input_file} not found")
            continue

        process_file(input_file)


if __name__ == "__main__":
    main()
