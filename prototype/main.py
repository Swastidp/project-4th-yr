import subprocess
import sys
from pathlib import Path


# Get the directory where this pipeline script is located.
BASE_DIR = Path(__file__).parent


# List of scripts executed sequentially to complete the F1 tyre strategy pipeline.
SCRIPTS = [
    "fetch_data.py",  # Fetch and prepare raw F1 data.
    "plot_degradation.py",  # Generate tyre degradation and pace-loss plots.
    "train_pace_loss.py",  # Train the pace-loss prediction model.
    "build_stints.py",  # Build and process tyre stint data.
    "prepare_km_web.py",  # Prepare processed data for the web interface.
    "reduce_chart_points.py",  # Reduce curves to important points for cleaner charts.
    "recommend.py",  # Generate tyre strategy recommendations.
]


def run_script(script):
    """
    Execute a single pipeline script.

    The script is run using the same Python interpreter and
    working directory as the main pipeline.
    """

    # Display which pipeline step is currently being executed.
    print("\n" + "=" * 70)
    print(f"RUNNING: {script}")
    print("=" * 70)

    # Execute the selected Python script as a separate process.
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / script)],
        cwd=BASE_DIR,
    )

    # Stop the entire pipeline if the script returns an error.
    if result.returncode != 0:
        print(f"\nFAILED: {script}")
        sys.exit(result.returncode)

    # Confirm that the current pipeline step completed successfully.
    print(f"\nCOMPLETED: {script}")


def main():
    """
    Run all stages of the F1 tyre strategy pipeline
    in the defined order.
    """

    # Display the pipeline title.
    print("=" * 70)
    print("F1 TYRE STRATEGY PIPELINE")
    print("=" * 70)

    # Execute every pipeline script sequentially.
    # If one script fails, run_script() stops the pipeline.
    for script in SCRIPTS:
        run_script(script)

    # Display a final success message after all scripts finish.
    print("\n" + "=" * 70)
    print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY")
    print("=" * 70)


# Start the pipeline only when this file is executed directly.
if __name__ == "__main__":
    main()
