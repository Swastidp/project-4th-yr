"""
F1 tyre strategy pipeline runner.

Executes every stage of the project in dependency order so the
whole thing can be produced with a single command.

Typical use:

    uv run main.py                # run every stage in order
    uv run main.py --list         # show the stages without running them
    uv run main.py --only recommend
    uv run main.py --from build_stints
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# Get the directory where this pipeline script is located.
BASE_DIR = Path(__file__).parent

# Directory that every stage reads from and writes to.
DATA_DIR = BASE_DIR / "data"


# Stages executed sequentially to complete the F1 tyre strategy pipeline.
#
# Each entry is (stage name, script file, what the stage does).
# The order matters: a stage may only depend on outputs produced by
# the stages listed above it.
STAGES = [
    (
        "fetch_data",
        "fetch_data.py",
        "Fetch and clean raw FastF1 lap and weather data.",
    ),
    (
        "plot_degradation",
        "plot_degradation.py",
        "Plot the raw tyre degradation curves.",
    ),
    (
        "train_pace_loss",
        "train_pace_loss.py",
        "Train the pace-loss models and export the per-compound curves.",
    ),
    (
        "build_stints",
        "build_stints.py",
        "Build the stint table and fit the survival models.",
    ),
    (
        "make_tyre_radar",
        "make_tyre_radar.py",
        "Score each compound and draw the tyre radar chart.",
    ),
    (
        "prepare_km_web",
        "prepare_km_web.py",
        "Prepare the Kaplan-Meier curves for the web interface.",
    ),
    (
        "reduce_chart_points",
        "reduce_chart_points.py",
        "Reduce the curves to their most important points for cleaner charts.",
    ),
    (
        "recommend",
        "recommend.py",
        "Generate the tyre strategy recommendations.",
    ),
]


# Scripts that are not part of the default run but can still be
# selected by name with --only.
EXTRA_STAGES = [
    (
        "simplify_curves",
        "simplify_curves.py",
        "Older curve simplifier, superseded by reduce_chart_points.",
    ),
]


# Every stage that can be named on the command line.
ALL_STAGES = STAGES + EXTRA_STAGES


def stage_names(stages):
    """Return just the names of the given stages."""

    return [name for name, _, _ in stages]


def lookup(name):
    """
    Find a single stage by name.

    Exits with a helpful message if the name is not a known stage.
    """

    for stage in ALL_STAGES:
        if stage[0] == name:
            return stage

    print(f"ERROR: unknown stage: {name}")
    print(f"Known stages: {', '.join(stage_names(ALL_STAGES))}")
    sys.exit(2)


def format_duration(seconds):
    """Render a duration in a short, readable form."""

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, seconds = divmod(int(seconds), 60)

    return f"{minutes}m{seconds:02d}s"


def list_stages():
    """Print the pipeline stages and what each one does."""

    print("Pipeline stages, in run order:\n")

    for index, (name, script, description) in enumerate(STAGES, start=1):
        print(f"  {index}. {name:<20} {description}")

    print("\nNot run by default (select with --only):\n")

    for name, script, description in EXTRA_STAGES:
        print(f"     {name:<20} {description}")

    print("\nRun a subset with --only / --skip / --from.")


def select_stages(args):
    """
    Work out which stages to run from the command line options.

    --only wins outright; otherwise the default list is trimmed by
    --from and then by --skip.
    """

    if args.only:
        return [lookup(name) for name in args.only]

    selected = STAGES

    # Drop everything before the requested starting stage.
    if args.start:
        names = stage_names(STAGES)

        if args.start not in names:
            lookup(args.start)  # Exits with the standard error message.

        selected = selected[names.index(args.start) :]

    # Drop any stage the caller asked to skip.
    if args.skip:
        for name in args.skip:
            lookup(name)

        selected = [stage for stage in selected if stage[0] not in args.skip]

    return selected


def run_stage(stage, index, total):
    """
    Execute a single pipeline stage.

    The script is run using the same Python interpreter and
    working directory as the main pipeline. Returns the elapsed
    time in seconds, or exits if the stage fails.
    """

    name, script, description = stage

    path = BASE_DIR / script

    if not path.exists():
        print(f"\nFAILED: {script} does not exist")
        sys.exit(1)

    # Display which pipeline step is currently being executed.
    print("\n" + "=" * 70)
    print(f"[{index}/{total}] RUNNING: {name}  ({script})")
    print(description)
    print("=" * 70)

    started = time.monotonic()

    # Execute the selected Python script as a separate process.
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=BASE_DIR,
    )

    elapsed = time.monotonic() - started

    # Stop the entire pipeline if the script returns an error.
    if result.returncode != 0:
        print(f"\nFAILED: {name} exited with code {result.returncode}")
        print(f"Re-run just this stage with: uv run main.py --only {name}")
        sys.exit(result.returncode)

    # Confirm that the current pipeline step completed successfully.
    print(f"\nCOMPLETED: {name} in {format_duration(elapsed)}")

    return elapsed


def parse_args(argv):
    """Define and parse the command line options."""

    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run the F1 tyre strategy pipeline end to end.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="show the pipeline stages and exit",
    )

    parser.add_argument(
        "--only",
        nargs="+",
        metavar="STAGE",
        help="run only these stages, in the order given",
    )

    parser.add_argument(
        "--skip",
        nargs="+",
        metavar="STAGE",
        help="run the full pipeline except these stages",
    )

    parser.add_argument(
        "--from",
        dest="start",
        metavar="STAGE",
        help="start at this stage and run everything after it",
    )

    return parser.parse_args(argv)


def main(argv=None):
    """
    Run the requested stages of the F1 tyre strategy pipeline
    in dependency order.
    """

    args = parse_args(argv)

    # Showing the stages is a request for information, not a run.
    if args.list:
        list_stages()
        return 0

    selected = select_stages(args)

    if not selected:
        print("Nothing to run: every stage was skipped.")
        return 0

    # Make sure the shared output directory exists before any stage
    # tries to write into it.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Display the pipeline title.
    print("=" * 70)
    print("F1 TYRE STRATEGY PIPELINE")
    print(f"{len(selected)} stage(s): {', '.join(stage_names(selected))}")
    print("=" * 70)

    started = time.monotonic()

    # Execute every selected stage in order.
    # If one stage fails, run_stage() stops the pipeline.
    for index, stage in enumerate(selected, start=1):
        run_stage(stage, index, len(selected))

    total = time.monotonic() - started

    # Display a final success message after all stages finish.
    print("\n" + "=" * 70)
    print(f"ALL STAGES COMPLETED SUCCESSFULLY in {format_duration(total)}")
    print(f"Outputs are in: {DATA_DIR}")
    print("=" * 70)

    return 0


# Start the pipeline only when this file is executed directly.
if __name__ == "__main__":
    sys.exit(main())
