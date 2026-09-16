# Mid-Sem Deck Content — Team T06 "DedSec Program"

Source material for all 16 required slides (per `DSC-23-27-DSC4195-Project-II-Instructions.pdf`),
submission deadline **2026-09-20, 11:59 PM** via the Google Form. Everything below is drawn from
real, run code in `prototype/` and the design doc — nothing here is invented. A few spots need a
teammate to fill in a blank (marked **[FILL IN]**) because the answer depends on things I don't
have: your Project I SMART criteria wording, who actually owns which piece, and mentor-confirmed
dates.

Copy/paste and trim to fit — this is source material, not final slide text.

---

## What's actually built (read this first)

A working, tested F1 tyre-strategy pipeline in `prototype/` (Python, `uv`-managed):

1. **Data pipeline** (`fetch_data.py`) — pulls real race data via FastF1 (open-source F1 telemetry
   library), caches it locally, cleans it with 3 verified filters. **9 real races, 8,745 clean
   laps.**
2. **Pace-loss model** (`train_pace_loss.py`) — predicts how many seconds a lap slows down as a
   tyre ages. **Beats a "guess the average" baseline by 20% on dry races** (0.76s error vs 0.95s
   baseline), tested honestly on races the model never saw during training.
3. **Stint-life model** (`build_stints.py`) — predicts how many laps a tyre lasts before it's
   changed, correctly handling races that end before the tyre "fails." **530 real stints**, results
   land within 1-2 laps of published reference numbers.
4. **Recommender** (`recommend.py`) — combines both models into an actual tyre/stint-length
   recommendation. Works correctly on the pit-stop math and wet-weather routing; has one honestly
   diagnosed, explained limitation (below).

Three charts exist and are ready to drop into slides:
- `prototype/data/degradation_curve.png` — raw pace-loss-vs-tyre-age chart (first attempt).
- `prototype/data/degradation_curve_corrected.png` — the fixed version, two panels (before/after
  fuel correction), this is the one to use.
- `prototype/data/survival_curves.png` — Kaplan-Meier stint-survival curves per tyre compound.

**The one open limitation, worth stating plainly rather than hiding:** the recommender doesn't
always rank the fast-wearing (soft) tyre correctly against the durable (hard) one for short
stints. Root cause, proven not to be a bug: F1's tyre rules mean the "hard" label means a
different physical tyre at different tracks, and with only 8 circuits in the sample, the model
can't fully separate "which tyre" from "which track." This needs either data that isn't public
(the true compound codes) or many more races than fit in the time available — it's a genuine,
provable data-scarcity limit, not a mistake. This is good report material, not something to bury.

---

## Slide 1 — Title

- Team **T06**, project name **"Race Strategy Optimisation"** (or update if you want a name that
  better reflects the tyre-degradation focus — current one still fits).
- Parantap Guha — Roll 2362005, Autonomy Roll 12623019035
- Ayush Giri — Roll 2362011, Autonomy Roll 12623019016
- Bishal Laha — Roll 2362022, Autonomy Roll 12623019021
- Swastidip Maji — Roll 2362023, Autonomy Roll 12623019059

---

## Slide 2 — Carry-forward and refinement

**Problem (restated):** Predict how an F1 tyre's performance degrades over a stint and how long it
will last, to recommend the right tyre and stint length — replacing reactive, experience-based
pit-stop decisions with a model grounded in real telemetry data.

**Target users/beneficiaries:** [FILL IN — restate from your Project I report; likely: race
strategists / motorsport analysts, with a longer-term extension to everyday tyre-safety
applications you pitched in Project I]

**Success criteria:** [FILL IN — paste your Project I SMART criteria here so slide 9's evaluation
plan can explicitly tie back to it, as the guideline requires]

**What changed since Project I, and why:** The original pitch implied a model that could pick
*which compound* to run. During the "prepare" phase we built and tested this directly — a
compound classifier scored only 45.8% accuracy against a 40.5% "just guess the most common
compound" baseline. That's not a tuning problem: which compound a team runs is a strategic,
rules-constrained choice (teams must use 2+ compound types, only 3 of 5 possible compounds are
even brought to each race), not something physics predicts. **We pivoted from "one model predicts
the compound" to a two-model system** (a pace-loss model + a stint-survival model) feeding a
deterministic scoring rule — the same architecture used in a published academic reference
(Heilmeier et al.'s "Virtual Strategy Engineer," TUM). This is a stronger, evidence-based project
than the original framing.

---

## Slide 3 — Prototype definition and scope

**Minimum viable prototype (delivered by end of 7th semester):**
- A reproducible data pipeline pulling and cleaning real F1 telemetry (done — 9 races).
- A validated pace-loss model showing measurable, honestly-tested improvement over a naive
  baseline (done — 20% better on dry-race data, held-out testing).
- A validated stint-length model handling censored data correctly (done — matches published
  reference numbers).
- A working (if imperfect) recommendation layer combining both, with documented limitations.

**Explicitly out of scope this semester, deferred to the 8th:**
- A production dashboard/UI (Streamlit or similar) — the design doc's own lowest-priority item.
- Resolving the tyre/circuit confound in the recommender — needs either non-public compound data
  or a much larger multi-season race sample.
- The originally-pitched cross-industry extension (trucking/daily-driver tyre safety) — kept as a
  stated future direction, not part of this semester's deliverable.
- Full 3-season-scale data (current: 9 races; doc's reference results used ~22 races/3 seasons).

**Definition of done (for the examiner to check against):** pipeline runs end-to-end from raw
FastF1 pull to a printed recommendation table; pace-loss model beats its baseline on held-out
races; stint model's accuracy score (C-index) beats random chance (0.5) by a wide margin; every
number quoted in the report is reproducible by re-running the scripts in `prototype/`.

---

## Slide 4 — System / solution design (block diagram)

Draw this as boxes and arrows:

```
FastF1 (live F1 timing data)
        |
Local cache (avoids 500-calls/hour rate limit)
        |
Cleaning filters: accuracy flag -> green-flag-only -> drop laps >5% slower than race median
        |
Feature engineering: tyre age, fuel-load estimate, weather join
        |
        +--------------------------+
        |                          |
Pace-loss model              Stint-life survival model
(Ridge regression)           (Kaplan-Meier + Cox regression)
        |                          |
        +------------+-------------+
                      |
        Recommendation scorer (rule-based, no training):
        cost = predicted pace loss + pit-stop time, per tyre/stint-length option
        wet conditions -> routed straight to intermediate tyre by rule
                      |
              Ranked output table
        (tyre choice + stint length + score)
```

## Slide 5 — Chosen approach, alternatives, justification (one row per block)

| Block | Chosen approach | Alternative considered | Why |
|---|---|---|---|
| Data source | FastF1 (open-source, free) | Proprietary F1 team telemetry | Not publicly accessible; FastF1 is the standard open tool used in published research on this exact problem |
| Compound recommendation | Two-model decomposition + rule-based scorer | Single classifier predicting compound directly | Tested directly: classifier scores 45.8% vs 40.5% baseline (fails); decomposition matches a published, working architecture |
| Stint-length prediction | Survival analysis (Kaplan-Meier + Cox) | Direct regression on stint length | Tested directly: direct regression scores essentially zero skill (R²≈0) because ~1/3 of stints don't "fail," they just run out of race — survival analysis handles this correctly |
| Pace-loss model | Ridge regression (linear) as primary | XGBoost (gradient-boosted trees) | Both were tested; XGBoost is usually stronger in the literature, but on our smaller sample it couldn't generalize to race circuits it hadn't seen, while Ridge could — an explainable, reportable finding, not an oversight |

---

## Slide 6 — Data plan (part 1: what was obtained)

- **Source:** FastF1 (wraps official F1 timing data), Python library.
- **Obtained, not just identified:** 9 real race weekends already pulled and cached (2024 Bahrain,
  Spanish, British, Canadian, Italian, Japanese, Austrian, Dutch Grands Prix; 2023 Italian GP).
- **Sample sizes:** 8,745 clean laps (pace-loss dataset); 530 tyre stints (survival dataset,
  built separately from raw, unfiltered laps because stint boundaries need every lap counted).
- **Schema:** per lap — driver, lap time, tyre compound, tyre age, track status, FastF1's own
  accuracy flag; joined weather — air/track temperature, humidity, rainfall flag.

## Slide 7 — Data plan (part 2: quality, preprocessing, split, ethics)

- **Known quality issues found and handled:** one driver's entire lap set was flagged unreliable
  by FastF1 at one race (caught, not silently dropped); FastF1's own fuzzy race-name matching once
  silently substituted the wrong race — caught by inspecting logs, fixed by using exact circuit
  names. FastF1's rate limit (500 calls/hour) is mitigated with a persistent local cache — verified
  a full re-run needs **zero** new network calls.
- **Preprocessing:** three-filter cleaning pipeline for the pace-loss model (drop
  unreliable/safety-car/traffic-affected laps — 88% of raw laps survive the first two filters,
  in line with expectations); stint boundaries computed from raw, unfiltered laps instead, since
  filtering would corrupt the true observed length of a stint.
- **Train/test split:** race-grouped cross-validation — each fold holds out one entire race, never
  mixing laps from the same race across train and test (a naive random split would leak
  information and inflate accuracy scores).
- **Consent/licensing/privacy:** FastF1 redistributes official, public F1 timing data under an
  open-source license; no private or personal data is involved (public sporting performance data
  only).

---

## Slide 8 — Methods and technology stack

- **Language/tooling:** Python, managed with `uv` (fast package manager).
- **Data:** FastF1 (telemetry), pandas (data handling).
- **Modelling:** scikit-learn (Ridge regression), XGBoost (gradient-boosted trees, comparison
  model), lifelines (Kaplan-Meier + Cox proportional-hazards survival analysis).
- **Compute:** local laptop — no GPU or cloud compute needed, dataset is small/tabular.
- **Comparison method:** Ridge vs. XGBoost compared via race-grouped cross-validation (MAE and R²
  on races neither model trained on); Ridge selected as the primary model because it generalizes
  better at this data scale — XGBoost is a stated candidate to revisit once more races are added.

---

## Slide 9 — Evaluation plan

- **Metrics:** Mean Absolute Error (MAE, in seconds) and R² for the pace-loss model; concordance
  index (C-index — how often the model correctly ranks which of two stints ends first) and hazard
  ratios for the stint-life model.
- **Baselines:** "predict the average" for pace loss (beaten by 20% on dry races); random chance
  (C-index 0.5) for stint survival (beaten: 0.77).
- **Validation protocol:** race-grouped cross-validation — train on some races, test on a race the
  model has genuinely never seen, rotate through all races. This is a stricter, more honest test
  than a random split.
- **Tie-back to Project I SMART criteria:** [FILL IN — paste your original success criteria and
  map explicitly: e.g. "criterion X ('predict tyre wear within N seconds') → achieved at 0.76s
  MAE on held-out races"]
- **Human/domain-expert judgment:** [FILL IN — if you plan any qualitative review, e.g. does the
  recommender's output match what a real strategist would choose; not yet done, could be a
  stretch goal for slide 13's dependencies]

---

## Slide 10 — Timeline and milestones

| Window | Milestone | Status |
|---|---|---|
| By Sept 16 | Data pipeline, pace-loss model, stint-life model, recommender — all built and validated | **Done** |
| Sept 16–19 | Assemble and polish the mid-sem deck; add any figures/wording gaps | In progress |
| Sept 20, 11:59 PM | Mid-sem submission | Deadline |
| [FILL IN — weeks after] | Expand race sample (target: repeat visits to same circuits under different conditions, to help separate the tyre/track confound noted on slide 12) | Planned |
| [FILL IN] | Investigate whether true Pirelli compound codes are obtainable anywhere, to fully resolve the recommender's known limitation | Planned, depends on mentor input (see slide 13) |
| [FILL IN] | Build the dashboard/UI layer (explicitly deferred, lowest priority) | Planned, stretch |
| End of 7th semester | End-semester evaluation — "definition of done" per slide 3 | Target |

*(Fill in real calendar dates for the "planned" rows — I don't have your semester calendar.)*

---

## Slide 11 — Task allocation

**[FILL IN with real names/ownership — this is a suggested split by component, not an actual
record of who did what; assign to match real contributions.]**

| Component | Suggested owner | Demonstrates at end-sem |
|---|---|---|
| Data pipeline (`fetch_data.py`) | ? | Live pull + cache + filter pipeline running end-to-end |
| Pace-loss model (`train_pace_loss.py`) | ? | Cross-validated accuracy result, explain Ridge-vs-XGBoost finding |
| Stint-life model (`build_stints.py`) | ? | Kaplan-Meier/Cox results, explain censoring |
| Recommender (`recommend.py`) + writeup | ? | Live recommendation demo, explain the diagnosed limitation |

---

## Slide 12 — Risk register (reviewed from Project I)

| Risk | Status | Mitigation | Owner |
|---|---|---|---|
| Data integrity / sensor reliability | **Materialized, mild** — one driver's laps wholesale-flagged unreliable at one race; a silent wrong-race substitution was caught by log inspection | Printed self-checks after every pipeline run; explicit per-race/per-driver row counts | [FILL IN] |
| Model overfitting | **Materialized** — recommender's tyre ranking is skewed by only having 8 circuits in the sample (tyre label confounded with track identity) | Diagnosed and partially corrected (closed ~40% of the gap); documented as needing more race diversity, not silently hidden | [FILL IN] |
| Trust gap / liability | Not yet tested — no end users evaluated the output yet | Deferred to a future user-feedback phase | [FILL IN] |
| Hardware/software latency | Receded for now (batch analysis only, no real-time requirement tested this semester) | N/A yet | — |
| **New: FastF1 API rate limit** | Active constraint (500 calls/hour) | Persistent local cache implemented and verified (zero network calls on re-run) | [FILL IN] |
| **New: wet-weather data scarcity** | Active — too few wet races for reliable wet-tyre modelling | Scoped explicitly as rule-based/descriptive only for wet conditions, not a supervised-learning claim | [FILL IN] |

---

## Slide 13 — Dependencies and clarifications

- **Ask:** Is the current "dry-conditions only" scope for the supervised models acceptable for
  mid-sem, given wet-race data is genuinely too scarce this semester? — **Who:** mentor —
  **By:** [FILL IN date]
- **Ask:** Is there any legitimate way to access the true Pirelli compound codes (C1–C5) behind
  FastF1's weekend-relative SOFT/MEDIUM/HARD labels, to fully resolve the recommender's diagnosed
  limitation? — **Who:** mentor / Pirelli-adjacent contact if any — **By:** [FILL IN date]
- **No compute or API-key dependency** — FastF1 is free and requires no authentication.

---

## Slide 14 — Preliminary evidence

- **Chart 1:** `prototype/data/degradation_curve_corrected.png` — before/after fuel correction;
  shows tyre wear becoming visible once fuel-burn is accounted for.
- **Chart 2:** `prototype/data/survival_curves.png` — Kaplan-Meier stint-survival curves, one per
  tyre compound.
- **One baseline number:** pace-loss model — 0.76 seconds average error, 20% better than guessing
  the average, on race data the model never trained on.
- **One more number:** stint-life model — correctly ranks which of two stints ends first 77% of
  the time (vs. 50% for a coin flip).
- Optional: a screenshot of `recommend.py`'s printed output table, showing a live tyre/stint-length
  recommendation for a sample race scenario.

---

## Slide 15 — Effort and cost estimate

- **Monetary cost: effectively $0** — FastF1 is free and open-source, no cloud compute or paid API
  used, ran entirely on a personal laptop. (Notably cheaper than the ~$150 estimated in the
  Project I deck, since no infrastructure spend was needed.)
- **Time:** [FILL IN actual hours — today's build alone covered: data pipeline setup, two model
  types, a survival analysis, a recommender, and a full debugging/validation cycle on each. Give
  an honest hour estimate per teammate/component rather than reusing the old deck's untested
  260-hour figure.]

---

## Slide 16 — References

Prefer these — they're the papers the actual technical approach is grounded in (from the design
doc), over the general F1-journalism references in the old deck:

1. Heilmeier, Thomaser, Graf & Betz (2020), "Virtual Strategy Engineer: Using Artificial Neural
   Networks for Making Race Strategy Decisions in Circuit Motorsport," *Applied Sciences* 10(21).
   — the architecture this project's two-model decomposition follows.
2. Todd et al. (2025), "Explainable Time Series Prediction of Tyre Energy in Formula One Race
   Strategy," arXiv:2501.04067. — justifies choosing gradient boosting/Ridge over deep learning.
3. "A State-Space Approach to Modeling Tire Degradation in Formula 1 Racing," arXiv:2512.00640. —
   closest methodological analogue, also FastF1-based; source of the fuel-correction approach.
4. Sulsters & Bekker, "Simulating Formula One Race Strategies." — source of the lap-time
   decomposition (base + fuel + tyre + noise) this project's target variable is built from.
5. (Optional, from the old deck) FIA 2026 Sporting Regulations / Pirelli compound-allocation
   articles — for the "why not a classifier" domain-grounding point on slide 2.

*(Full list with links is in `FastF1 Tyre Compound & Stint-Life Prediction .../fastf1-tyre-project-plan.md`, References section.)*
