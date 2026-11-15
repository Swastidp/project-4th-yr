# The Code, Explained in Plain English

A walkthrough of every script and every function in `prototype/`, what it does, why it
exists, and how the pieces fit together. Written for the presentation — no jargon
unless it is explained first.

---

## 1. The one-line story

> **We take real Formula 1 race data, measure how much slower a car gets as its tyres
> age, measure how long tyres actually stay on the car, and then combine those two
> measurements into a recommendation: "fit this tyre, run it for this many laps."**

Everything in the repo is one of four things: **getting data**, **modelling pace**,
**modelling tyre life**, or **turning those into advice**.

---

## 2. Why the project is shaped this way (the key idea)

The obvious approach would be: train a classifier that outputs "SOFT", and a regressor
that outputs "18 laps". The design doc shows this was tried and it **fails** —
45.8% accuracy against a 40.5% baseline, i.e. almost no skill.

The reason is not bad tuning, it is the wrong question:

- **Which tyre a team fits is a human strategy decision**, constrained by F1 rules
  (you *must* use two compounds; only 3 of 5 compounds are available each weekend).
  A model trained on that learns the rulebook, not tyre physics.
- **When a stint ends is a pit-wall decision**, or the chequered flag, or a crash —
  not the tyre wearing out.

So the project **splits the problem into two things that are physically measurable**:

| Stage | Question asked | Why it is answerable |
|---|---|---|
| Stage 1 | How much slower is a lap when the tyre is *n* laps old? | Lap times are measured directly |
| Stage 2 | How long does a stint survive? | Survival analysis handles "we don't know when it *would* have died" |
| Stage 3 | Given both, what is the cheapest plan in seconds? | Just arithmetic — no new model |

This is the same decomposition used by TUM's "Virtual Strategy Engineer" paper.

---

## 3. The pipeline at a glance

```
1 ... fetch_data.py    →  data/laps_clean.csv      (14,454 clean laps, 16 races)
      │
      ├── 2 ... plot_degradation.py   →  degradation_curve.png    (the "before" picture)
      │
      ├── 3 ... train_pace_loss.py    →  degradation_curve_corrected.png
      │                                  before_*.csv / after_*.csv (Stage 1: pace model)
      │
      └── 4 ... build_stints.py       →  stints.csv, km_survival.csv
                                         survival_curves.png        (Stage 2: life model)
                    │
                    ├── 5 ... make_tyre_radar.py     →  tyre_radar.png + scores
                    ├── 6 ... prepare_km_web.py      →  kaplan_meier_web.csv
                    ├── 7 ... reduce_chart_points.py →  main_*.csv   (slim charts)
                    └── 8 ... recommend.py           →  the ranked recommendation
```

`main.py` runs all 8 stages in that order with one command, in about 53 seconds.

---

## 4. `main.py` — the conductor

**Why it exists:** eight scripts have to run in a fixed order, because each one reads the
file the one before it wrote. This removes the chance of running them in the wrong order.
It can also run just part of the pipeline, so you don't wait for everything after a small change.

| Function / constant | What it does, simply |
|---|---|
| `STAGES` | The running order, written once. Each entry is (stage name, script file, one-line description). Adding a stage = adding an entry. |
| `EXTRA_STAGES` | Scripts that are **not** run by default. Today that's only `simplify_curves`, the old version of `reduce_chart_points`. |
| `parse_args()` | Reads the command-line options: `--list`, `--only`, `--skip`, `--from`. |
| `list_stages()` | Prints the stages and what each one does (`--list`). Nothing runs. |
| `select_stages(args)` | Works out what to run. `--only` wins outright; otherwise it starts from `--from` and removes anything in `--skip`. |
| `lookup(name)` | Finds a stage by name. A typo stops the run with the list of real names. |
| `run_stage(stage, index, total)` | Runs one script as a separate program and times it. **If the script fails, the whole pipeline stops** and prints the exact command to re-run only that stage. |
| `format_duration(seconds)` | Turns seconds into "12.3s" or "1m05s" for the progress lines. |
| `main()` | Puts it together: parse options, pick stages, make sure `data/` exists, run each stage, print the total time. |

Handy commands:

```bash
uv run main.py                         # everything
uv run main.py --list                  # what are the stages?
uv run main.py --only recommend        # just one stage
uv run main.py --from build_stints     # this stage and everything after
```

**Presentation line:** *"One command, `uv run main.py`, reproduces every number and every
chart in this report from scratch, in under a minute."*

---

## 5. `fetch_data.py` — getting honest data

This is the foundation. The design doc makes the point bluntly: **lap filtering affects
quality more than any model setting**.

| Function / constant | What it does, simply |
|---|---|
| `RACES` | The 16 races we study: 7 from 2022, Monza 2023, and 8 from 2024, on 8 circuits. Chosen for variety: hot and dry (Bahrain), low-degradation (Monza), wet (Silverstone, Canada, Suzuka 2022). Adding a race is one line. |
| Excluded on purpose | **2022 Austria** is commented out. Its source tyre data is broken: 250 of 1,324 laps have no compound. The other 2023 races and all of 2025 are blocked, because since October 2026 F1's timing server needs an F1TV login. |
| `PIRELLI_NOMINATIONS` | New. The **real** compound behind each weekend's HARD / MEDIUM / SOFT label, for 32 races, each line with its source link. Example: at Bahrain 2024, HARD = C1; at Monza 2024, HARD = C3. Same label, different rubber. |
| `compound_code(year, event, compound)` | New. Looks up a lap's real code (C0–C5). In 2023 Pirelli renamed its old C1 to "C0", so a 2022 "C1" is stored as "C0". That way the same name always means the same rubber. Returns nothing for INTERMEDIATE and WET. |
| `EXPECTED_EVENT_NAMES` | New. The race name each entry *should* load. FastF1 matches names loosely and once silently loaded the wrong race. Now a mismatch stops the run. |
| `LAP_COLS`, `WEATHER_COLS` | The columns we care about — tyre compound, tyre age, lap time, plus temperature, humidity, rainfall. |
| `load_race(year, event)` | Does the real work for **one** race (details below). |
| `main()` | Loops over all races, stacks them into one table, saves `laps_clean.csv`, then prints a **self-check summary** and runs `assert` checks. It also checks every race has a Pirelli entry. |

### Inside `load_race` — the four steps that matter

1. **Load the session** from FastF1 (an open-source library that serves official F1
   timing data), with `telemetry=False, messages=False` so it loads fast, and using a
   **local cache** — the F1 API allows only 500 calls/hour, so caching is mandatory.
2. **Merge the weather onto the laps.** Weather is recorded roughly once a minute, laps
   happen roughly every 90 seconds, so the timestamps never match exactly.
   `pd.merge_asof(..., direction="nearest")` attaches each lap to the *closest in time*
   weather reading. A normal join would simply lose most rows.
3. **Three filters, in order** — this is the quality gate:
   - `IsAccurate` — FastF1's own "this lap's timing is trustworthy" flag.
   - `TrackStatus == "1"` — green flag only. Removes Safety Car and VSC laps, where
     everybody is slow for reasons that have nothing to do with tyres.
   - `LapTime < 1.05 × race median` — removes laps ruined by traffic, and the
     in-lap/out-lap around a pit stop.
   - New: laps with no compound or no tyre age are dropped. No tyre model can use them.
4. **Check it's the right race**, then **tag each lap** with its real compound code
   (`CCode` column), convert lap time to seconds, and add the year and race name.

The function returns both the cleaned laps **and the row counts after each filter**, so
`main()` can print a table showing exactly how many laps survived each stage.
On the current 16 races, filters 1+2 keep 84.3% of raw laps, and all filters together keep
82% (14,454 of 17,678).

**Presentation line:** *"We don't just say the data is clean — the script prints the
survival rate of every filter on every race, every run."*

---

## 6. `plot_degradation.py` — the "before" picture

A deliberately simple chart, and it is included **because it looks wrong**.

| Function | What it does, simply |
|---|---|
| `main()` | Computes **pace loss** = each lap's time minus that driver's own best lap in that race. Averages it per (compound, tyre age). Throws away any tyre-age point backed by fewer than `MIN_LAPS = 20` laps, so noisy points don't create fake trends. Plots one line per compound and prints an early-vs-late trend summary. |

**Why "pace loss vs the driver's own best lap"?** It cancels out, in one step, how fast
the car is, how good the driver is, and how long the circuit is. What is left is mostly
tyre age, fuel load and weather.

**The punchline:** the raw curves often show lap times getting *faster* as tyres age.
That is not magic — **the car is burning fuel and getting lighter** (about −0.069 s/lap),
which hides tyre wear completely. This chart sets up the problem that Stage 1 solves.

---

## 7. `train_pace_loss.py` — Stage 1, the pace model

The core modelling script. It trains **three models on the same target**: a Ridge
regression for *explaining*, the same Ridge grouped by **real compound code** (new), and
XGBoost for *accuracy*.

| Function | What it does, simply |
|---|---|
| `add_features(df)` | Builds what the models need. **`Delta`** = the target (lap time − driver's own best lap that race). **`FuelPct`** = `1 − LapNumber/TotalLaps`, i.e. how full the tank is — this is how we separate fuel effect from tyre effect. **`Race`** = a unique race ID used for honest testing. New: **`TyreKey`** = the lap's real compound code (C0–C5), or the label for INTERMEDIATE / WET, which have no code. |
| `ridge_design(df, compounds, key="Compound")` | Builds the table of inputs for Ridge, **written out by hand on purpose** so every number is interpretable. For each compound it creates its own baseline, its own "seconds lost per lap of age", and its own curvature (`age²`) term; then two shared columns, fuel and track temperature. The `age²/100` scaling just keeps the numbers a readable size. New: `key` picks what to group by. The default is the SOFT/MEDIUM/HARD label, so every older caller works unchanged; `key="TyreKey"` groups by real code instead. |
| `make_ridge()` | Ridge regression with `fit_intercept=False` — because the per-compound baseline columns already play that role. |
| `make_xgb()` | XGBoost, the gradient-boosting model, used purely as the accuracy benchmark. |
| `cv_scores(...)` | **The honesty engine.** Uses `GroupKFold` grouped by race: an entire race is held out, the model trains on the other races and is tested on the unseen one. This prevents the model from memorising a circuit. It also scores a **baseline** ("always guess the median") on the same split, so we can prove the model adds skill rather than just looking good. |
| `print_folds(name, folds)` | Prints a per-race results table with a `beats?` column — yes/NO against the baseline, race by race. |
| `make_plot(...)` | The two-panel **before/after** figure. Left = raw averages (fuel still hiding the wear). Right = the Ridge curves with fuel and track temperature **held at their average value**, so the only thing changing is tyre age. The right panel is the corrected picture. |
| `export_chart_csvs(...)` | Writes `before_<compound>.csv` and `after_<compound>.csv` so the same curves can be redrawn in the dashboard/slides without re-running the model. |
| `main()` | Orchestrates: load → build both input tables → flag wet races → cross-validate → refit → report coefficients → save plots and CSVs → print PASS/FAIL checks. |

### Three things in `main()` worth calling out on a slide

- **Wet-race detection.** Any race with rainfall on more than 10% of its clean laps is
  labelled WET, and the models are scored twice: on everything ("pooled") and on dry
  races only. Wet and dry are different physics; mixing them silently would be dishonest.
- **The full-data refit.** Cross-validation measures *performance*; the model is then
  refit on all data purely to get **quotable coefficients** — e.g. how many seconds per
  lap each compound loses, the fuel effect across a full tank, seconds per °C of track
  temperature.
- **`net s/lap`.** Because the curve is quadratic, there is no single slope. This is the
  average gradient across the age range that compound is actually seen at (1 to the 98th
  percentile) — one fair, comparable number per compound.
- **Label vs real code (new).** Both Ridge versions go through the exact same
  hold-one-race-out test, and the script prints their dry-race errors side by side.

### Current results (16 races, run on 15 Nov 2026)

MAE = mean absolute error, the average miss per lap in seconds. Lower is better.

| Model | Dry races (13) | All races (16) |
|---|---|---|
| Ridge, by label | **0.763 s** | 1.531 s |
| Ridge, by real code | 0.786 s | 1.560 s |
| XGBoost | 0.902 s | 1.919 s |
| Baseline (always guess the median) | 0.981 s | 1.389 s |

- On dry races Ridge is 22% better than guessing, and beats the baseline on 12 of 13 races.
- On all races every model loses to the baseline. One race causes most of it: 2022 Japan
  (87% rain), where Ridge misses by 8.2 s a lap. Those 3 checks print FAIL on purpose.
- The real codes did **not** make the model more accurate. They did help the recommender
  (section 11).

**Presentation line:** *"Ridge tells us why, XGBoost tells us how well, and every number
is measured on races the model has never seen."*

---

## 8. `build_stints.py` — Stage 2, how long a tyre lasts

Conceptually the most interesting script, and the project's novelty angle.

### The problem it solves: censoring

A stint (one set of tyres, start to finish) ends for three different reasons:
the team pitted, the race finished, or the driver retired. Only the first tells you
anything about the tyre. In our 16 races, about 1 in 3 stints (318 of 1,000) are still
running when the race ends — we know the tyre lasted *at least* that long, but not how long it *would* have
lasted. That is called **right-censoring**, and it is exactly what **survival analysis**
(the same maths used for medical trials) is built for. Ignoring it makes every tyre look
worse than it is.

### It deliberately does NOT use `laps_clean.csv`

Important detail for the viva. Stage 1's filters remove pit laps, Safety Car laps and
slow laps — perfect for measuring pace, but it would make stints look **shorter than
they really were**. So this script re-reads the **raw** session laps from the local
FastF1 cache, with `offline_mode(True)` so it can never silently download different data.

| Function | What it does, simply |
|---|---|
| `stints_for_race(year, event)` | Turns raw laps into **one row per driver, per stint**: how many laps it lasted, when it started and ended, which compound, whether the tyres were already used, plus weather matched to the stint start (`merge_asof` again) and the race's total lap count. |
| `add_censoring(stints)` | Sets the survival flag. **`event = 1`** if another stint followed (so we saw it end); **`event = 0`** if it was the driver's last stint (censored — endpoint unknown). Also records whether that last stint actually reached the chequered flag. |
| `km_table(stints, compounds)` | Fits a **Kaplan-Meier** curve per compound. KM answers "what is the probability a stint is still running after *n* laps?" while correctly accounting for censored rows. Returns both the summary numbers and the fitted curve objects. |
| `export_km_csv(fitters)` | Writes the survival probabilities **and their 95% confidence intervals** to `km_survival.csv`, so the chart can show uncertainty, not just a line. |
| `make_plot(fitters)` | Draws the Kaplan-Meier survival curves, dark-themed, with each compound in its real F1 colour (soft red, medium yellow, hard white, inter green). |
| `main()` | Loads all races from cache, combines, censors, validates, saves `stints.csv`, prints the KM table, fits the **Cox model**, exports, and runs PASS/FAIL checks. |

### The Cox proportional-hazards model (in `main()`)

Kaplan-Meier compares compounds one at a time. **Cox** asks: holding weather, race
length and starting lap constant, how much more likely is a stint to end right now on a
soft than on a hard? It outputs a **hazard ratio**. Today SOFT's is 4.25: a soft stint is
about four times as likely to end on any given lap as a hard one. HARD is the baseline.
`C-index` is the quality score: how often the model correctly says which of two stints
ends first. 0.5 means random guessing; ours is 0.693 on 1,000 stints.

Practical safeguards in the code: compounds with fewer than 20 stints are excluded from
the fit (too few to trust), a covariate with no variation is dropped automatically, and a
convergence failure exits loudly instead of reporting nonsense.

**Bug fixed this semester.** The model only has on/off columns for SOFT, MEDIUM and
INTERMEDIATE; anything with all of them "off" is read as HARD. With 16 races, full-wet
tyres reached 21 stints, enough to pass the 20-stint minimum. They would have been
silently counted as HARD. Now only HARD, MEDIUM, SOFT and INTERMEDIATE go into the Cox
model, and anything else is printed as excluded.

### The honesty note that should be on the slide

The code says it in two places and so should you:

> The event being modelled is *"the team changed tyres"*, not *"the tyre failed"*.
> Hazard ratios describe strategy plus wear, not pure physics.

**Presentation line:** *"A third of stints never ended — the race did. Survival analysis is
the only correct way to handle that, and it's what makes this project methodologically
different from the usual F1 ML project."*

---

## 9. `prepare_km_web.py` — reshaping for the dashboard

Small and single-purpose.

| Function | What it does, simply |
|---|---|
| `main()` | Takes `stints.csv` and keeps just three columns, renamed to the standard survival-chart names: `stint_length → Time`, `event → Event`, `Compound → Group`. Then it **validates**: Event must be strictly 0 or 1, Time must be positive, blank group labels are dropped, and it errors loudly if nothing survives. |

**Why it exists:** web charting libraries expect `Time / Event / Group`. This keeps that
formatting concern out of the analysis script, and the validation means bad rows are
caught here rather than showing up as a silently wrong chart.

---

## 10. `reduce_chart_points.py` — making curves slide-friendly

A 40-point line is unreadable on a projector. This reduces each curve to **6 points that
keep its shape**.

| Function | What it does, simply |
|---|---|
| `point_distance(point, start, end)` | Measures how far a point sits *above or below* the straight line drawn between two other points. Big distance = that point is where the curve actually bends. |
| `simplify_curve(df, target_points)` | A greedy shape-preserving reducer. Keep the first and last points. Then repeatedly find the point that deviates most from the current simplified line and add it. Repeat until 6 points are kept. Corners survive; boring straight sections get dropped. (This is the Ramer–Douglas–Peucker idea.) |
| `process_file(input_file)` | Loads one curve CSV, checks the required columns exist, converts to numbers, sorts by tyre age, simplifies, and saves it as `main_<filename>.csv`. Skips missing or empty files with a message instead of crashing. |
| `main()` | Runs that over all 8 curve files (before/after × 4 compounds). |

> **Note for the team:** `simplify_curves.py` is an **earlier version of this same
> file**, kept for reference. `reduce_chart_points.py` is the one in the default run, and
> it adds the input validation and error handling. `simplify_curves.py` only runs if you
> ask for it: `uv run main.py --only simplify_curves`.

---

## 11. `recommend.py` — Stage 3, the actual answer

**No new model is trained here.** It reuses Stage 1's Ridge curve and Stage 2's
Kaplan-Meier medians, and does arithmetic. That is the whole point — it is transparent
and cheap.

### The cost function

```
Cost(compound, L)  =  total predicted time lost to tyre wear over L laps
                   +  PIT_LOSS × (remaining laps ÷ L)
```

In words: **running a tyre longer costs you pace, but pitting more often costs you pit
stops.** The best plan is the one where those two costs together are smallest. The second
term is the clever bit: if you have 52 laps left and stint for 20, you need about 2.6 more
stops, so you pay 2.6 × 22 seconds.

| Function | What it does, simply |
|---|---|
| `load_laps()` | Reloads `laps_clean.csv` and rebuilds `Delta` and `FuelPct` using **the exact same `add_features` function** as Stage 1 — so the recommender cannot drift from the model it claims to use. |
| `fit_ridge(df, key)` | Refits the Stage-1 Ridge on all data, but **drops the TrackTemp column** — see below. New: `key` picks the label model (default) or the real-code model (`"TyreKey"`). |
| `recommendation_design(df, compounds, key)` | Builds the prediction inputs the same way, with TrackTemp forced out. Keeps training and prediction perfectly consistent. |
| `km_medians()` | Pulls the Kaplan-Meier median stint length per dry compound from Stage 2 — used as a reality check on the recommendation. |
| `fitted(...)` | "Predict pace loss for this compound at these tyre ages with this fuel load." The small prediction helper everything else uses. |
| `cumulative_loss(...)` | Adds up predicted pace loss for ages 1, 2, 3 … L. Crucially it **lets fuel drop as the stint progresses** (the car gets lighter lap by lap), instead of pretending fuel is frozen. |
| `cost_table(...)` | Evaluates **every** (compound × stint length) combination, splits out `deg_cost` and `pit_cost` so you can see the trade-off, attaches the KM median, and raises a flag when a suggested stint is more than 1.5× longer than any real stint of that compound has been. Sorted cheapest first. |
| `recommend(...)` | The public entry point. **If it is raining, it returns INTERMEDIATE by rule** and skips the dry maths entirely. Otherwise it returns the ranked table. New: give it `nomination` (the race's real codes, e.g. HARD = C1, MEDIUM = C2, SOFT = C3) and it predicts each tyre from its real rubber, not its label. An unknown code stops with a clear error. |
| `show(title, table)` | Prints a readable top-8 table plus the best stint length per compound. |
| `compound_coefs`, `print_coef_table` | Diagnostics — prints each compound's fitted baseline, age slope, curvature and net gradient, so you can see what the recommendation is built on. |
| `main()` | Runs three demo scenarios — **lap 5 of 57** (long race ahead), **lap 45 of 57** (short run to the flag), **rainfall = True** — then runs 8 self-checks. New: it re-runs the first two with the real-code model, using the 2024 Bahrain codes, and prints both models' pick for the short run. |

### What the real codes changed (15 Nov 2026 run)

| Scenario | Label model | Real-code model |
|---|---|---|
| Lap 5, 52 laps left | HARD for 24 laps | SOFT for 23 laps, HARD only 0.1 s behind |
| Lap 45, 12 laps left | HARD; SOFT 4.0 s behind | HARD; SOFT **0.9 s** behind |

With 12 laps to go, a real strategist would usually pick SOFT. The label model was 4 s
off; with real codes the gap shrinks to under 1 s but doesn't flip. So most of the error
was "which track" being mixed up with "which tyre". More races should close the rest.

### Why rain is a rule, not a model

Only a few hundred wet laps exist in the whole dataset, and the FIA can *mandate* wet
tyres — so it is not a free strategic choice. A hand-written rule here is more honest than
a model trained on almost no data, and the code says so in its own docstring.

### Why TrackTemp is deliberately switched off

Track temperature *is* in the Stage-1 model and *is* reported there. But the recommender
zeroes it out, because with only 16 races it cannot reliably tell "this is a hot track"
apart from "this is Bahrain". Rather than claim temperature-awareness it doesn't have,
the code removes it and prints that limitation out loud.

### The self-checks at the bottom

Eight assertions run on every execution: model fitted, table non-empty, all costs finite
and positive, rain routes to INTERMEDIATE, rain skips dry optimisation, the long-stint
warning *does* fire when pushed to an absurd 200-lap race, and the warning *stays quiet*
for a sensible 5-lap stint. That last pair is a proper test — it checks the warning works
**and** that it isn't firing spuriously.

---

## 12. `make_tyre_radar.py` — the comparison chart

A comparison chart that scores all four compounds on five axes and draws a radar/spider
chart. It is stage 5 of the pipeline, so it refreshes on every full run.

| Function | What it does, simply |
|---|---|
| `validate_columns(df, required, name)` | Fails fast with a clear message if an input file is missing a column. |
| `normalize_mean_reference(values, higher_is_better)` | Puts five very different metrics (seconds, laps, s/lap) onto one comparable 0–100 scale. **50 = the average of the four compounds; 20 points = one standard deviation.** The `higher_is_better` switch flips the direction, so for "pace loss" a *lower* second count becomes a *higher* score. Every axis then reads "further out = better". |
| `fit_pace_model(laps)` | Fits the same Ridge pace model, reusing Stage 1's `ridge_design` so the radar and the recommender agree. |
| `predict_at_age(...)` | Predicts pace loss for one compound at one tyre age under fixed reference conditions. |
| `calculate_pace(...)` | Axis 1 — predicted pace loss at **tyre age 5** (settled, representative running). |
| `calculate_freshness(...)` | Axis 2 — predicted pace loss at **tyre age 1** (one-lap, brand-new performance). |
| `calculate_consistency(laps)` | Axis 3 — the standard deviation of `Delta`. Low spread = predictable tyre. |
| `calculate_degradation(...)` | Axis 4 — the fitted gradient from the 2nd to the 98th percentile of that compound's observed ages, i.e. how fast it falls away. |
| `calculate_longevity(stints)` | Axis 5 — median observed stint length, taken from Stage 2's `stints.csv`. |
| `main()` | Loads and cleans both inputs, fits the model **once**, computes all five metrics, merges them, normalises them, saves the raw values *and* the scores *and* the PNG. |

**Fairness detail:** every compound is compared at the **same** median fuel level and
**same** median track temperature, so the chart compares tyres, not circumstances.

**Built-in warning:** if INTERMEDIATE comes out with a *negative* degradation gradient
(the fit saying it gets faster with age, which is physically wrong and just reflects thin
wet data), the script prints a loud banner telling you not to read that as "inters resist
wear better". A model that warns you about its own weak spot is a strong thing to show.

---

## 13. How it all links together — the five-second version

1. **`fetch_data.py`** gets trustworthy laps and tags each with its real compound code →
   everything downstream depends on this.
2. **`plot_degradation.py`** shows the raw picture is misleading → motivates Stage 1.
3. **`train_pace_loss.py`** removes the fuel effect and produces the **cost per lap of
   running a tyre**.
4. **`build_stints.py`** produces the **realistic length of a stint**, correctly handling
   stints the race ended.
5. **`recommend.py`** combines "cost per lap" with "how often you'd have to pit" and
   ranks every option. **`make_tyre_radar.py`** summarises the same models visually.
6. **`main.py`** makes the whole thing one reproducible command.

### The three themes a marker will reward

- **Reformulation over brute force.** The headline result is that the obvious framing
  fails and a two-stage decomposition works.
- **Honest validation everywhere.** Race-grouped cross-validation, a median baseline to
  beat on every fold, PASS/FAIL check blocks at the end of every script, and filter
  survival rates printed every run.
- **Limitations stated, not hidden.** The 22-second pit loss is an assumption. Track
  temperature is switched off in the recommender. Rain is a rule. The stint "event" means
  *pitted*, not *worn out*. Every one of these is printed by the code itself.
