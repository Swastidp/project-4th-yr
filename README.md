# FastF1 Tyre Compound & Stint-Life Prediction

Predicts tyre pace loss, stint survival, and compound choice from FastF1 race
data. The whole thing runs as one pipeline.

## Run it

```bash
cd prototype
uv run main.py
```

That runs every stage in order and writes all outputs to `prototype/data/`.

The first run downloads race data from the FastF1 API and takes a minute or so.
Later runs reuse the cache in `prototype/.fastf1_cache/` and finish in about 15
seconds.

### Prerequisites

[uv](https://docs.astral.sh/uv/) — it creates the virtual environment and
installs the locked dependencies on first run, so nothing else needs setting up:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On macOS you also need the OpenMP runtime. xgboost links against it, and it is
a system library rather than a Python package, so uv cannot install it:

```bash
brew install libomp
```

Without it the pipeline fails at `train_pace_loss` with a `libxgboost.dylib
could not be loaded` error.

## Run part of it

```bash
uv run main.py --list                   # show the stages and what each one does
uv run main.py --only recommend         # re-run a single stage
uv run main.py --only train_pace_loss build_stints
uv run main.py --from build_stints      # this stage and everything after it
uv run main.py --skip fetch_data        # everything except this stage
```

Stage names come from `--list`. A failing stage stops the pipeline and prints
the command to re-run just that stage.

## Pipeline stages

Each stage depends only on outputs written by the stages above it.

| # | Stage | Produces |
|---|-------|----------|
| 1 | `fetch_data` | `laps_clean.csv` |
| 2 | `plot_degradation` | `degradation_curve.png` |
| 3 | `train_pace_loss` | `before_*.csv`, `after_*.csv`, `degradation_curve_corrected.png` |
| 4 | `build_stints` | `stints.csv`, `km_survival.csv`, `survival_curves.png` |
| 5 | `make_tyre_radar` | `tyre_radar_raw.csv`, `tyre_radar_scores.csv`, `tyre_radar.png` |
| 6 | `prepare_km_web` | `kaplan_meier_web.csv` |
| 7 | `reduce_chart_points` | `main_before_*.csv`, `main_after_*.csv` |
| 8 | `recommend` | strategy recommendations (printed) |

`simplify_curves` is an older version of `reduce_chart_points` kept for
reference. It is not part of the default run; select it with
`uv run main.py --only simplify_curves`.

## Layout

```
prototype/
  main.py          pipeline runner - stage list and ordering live here
  *.py             one file per stage
  data/            all generated CSVs and PNGs
  .fastf1_cache/   cached FastF1 API responses
```

To add or reorder a stage, edit the `STAGES` list in
[prototype/main.py](prototype/main.py).
