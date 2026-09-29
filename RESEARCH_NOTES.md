# Backtest Research Notes

Running log of things noticed while backtesting that could clue us into what
actually works. Add to this every time a backtest turns up something
non-obvious — a pattern, a data quirk, a design mistake that skewed results.
Don't log routine run output here, only things worth remembering next time.

---

## Scanner config FROZEN as v1 (2026-09-27)

`ScannerConfig` now carries `version="v1"` (`session_tracker.py`). These
values are locked: `approach_ticks=40, touch_ticks=8, hold_confirm_bars=3,
confluence_tolerance_ticks=8, volatility_percentile=0.60,
vwap_extension_atr_mult=0.5, momentum_expansion_bars=2`, plus
`SessionConfig`'s defaults (Asia 18:00-00:00 ET, London 00:00-09:00 ET,
NY 08:00-17:00 ET). **Do not retune any of these based on forward-outcome
(MFE/MAE) data** — the episode review below is for judging whether the
scanner is structurally reasonable, not for backfitting thresholds to
what happened afterward. A deliberate retune is a new tagged version
(v2+), never a silent edit to v1's numbers, so every `scan_runs` row stays
interpretable against the config that actually produced it.

## Opportunity Scanner — Phase 1 (2026-09-27)

Implemented and verified the deterministic (no LLM) Opportunity Scanner
architecture: `session_config.py` (explicit, configurable session windows
— Asia frozen at midnight ET, London begins immediately after with no gap,
overlapping sessions preserved as a set not a single label), `session_tracker.py`
(streaming PDH/PDL, Asia H/L, developing London H/L, VWAP, weekly pivots,
plus a 7-state per-level state machine: far/approaching/testing/breaking/
broken_holding/rejecting/reclaimed/lost), `opportunity_scanner.py` (the
`is_opportunity` rule: direct level interaction always counts;
"approaching + context" only counts with a supporting signal or confluence
— approaching alone never fires; four standalone signals — momentum
reversal/expansion, volatility expansion, VWAP extension — count on their
own regardless of levels), `run_scanner.py`, and `experiment_store.py`'s
new `scan_runs`/`opportunities` tables. `test_no_lookahead_scanner.py`
(3/3 passing) proves streaming session state can't leak future bars.

- **Two real bugs found while first testing this, not just threshold
  tuning.** (1) VWAP was included as a discrete level in the state
  machine — price oscillates through it constantly by definition, so
  `level_interaction` fired on ~95% of scans. Fixed by excluding VWAP from
  the level list entirely (it already has its own dedicated
  `vwap_extension` signal). (2) The ATR used for `vwap_extension` was a
  14-*minute* window — comparing VWAP's distance (a session-cumulative
  average) against an hour-scale-or-less volatility measure is a scale
  mismatch; median `|price-vwap|/atr` was 5.2x, so the 0.5x threshold was
  satisfied on ~95% of scans. Fixed with a separate `daily_atr` series
  (Wilder ATR on completed Globex daily bars, merge_asof-aligned, same
  no-lookahead pattern as `mtf_indicators.py`) — `vwap_extension` dropped
  to 4.8% of scans. (3) Design bug, not threshold: a level's `_LevelWatch`
  didn't reset to "far" when its price went back to `None` (e.g. a new
  Globex day resetting Asia/London) — it stayed frozen in whatever state
  it was in the day before, producing nonsensical reasoning like
  "AsiaHigh is testing" while `price=None`. Fixed by resetting state on
  every `None`-price update.

- **Fire-rate result, 13-week NQ window (2025-09-22 to 2025-12-19, the
  only reliable-density stretch of NQZ5 — see gappy-data note below):**
  5,819 scans, 5,201 (89.4%) raw `is_opportunity=True` — but only **85
  (1.5%) `is_new_episode=True`**. The raw rate looks high in isolation,
  but that's expected/intentional per "favor recall" — `level_interaction`
  alone is 78% of scans because once a level breaks and holds, it stays in
  an interaction state (by design) until price actually moves away or
  reverses through it again, which for at least one of 13 tracked levels
  is true most of the time over a whole week. The `is_new_episode` flag is
  the actually-actionable signal (≈1.3 new situations/trading day over 13
  weeks) — Phase 2 should gate on that, not on raw `is_opportunity`.
  Manually spot-checked the first 12 new-episode records: reasoning reads
  coherent and sensible (London high/low interaction narrative during
  London hours, transitioning to NY-session momentum signals, session
  labels correctly track the London→NY handoff).

- **One likely-gappy-data artifact, not a new scanner bug**: the very
  first scan of the run showed `WeeklyR1 is lost (-227.08pts)` — a level
  227 points away shouldn't still read as an interaction state (should
  have reset to "far" well before that distance). This is consistent with
  the already-documented NQZ5 gappy-data issue (see Data Quality below) —
  the scanner's warmup period replays from the start of the file, which
  includes the unreliable pre-Sept-22 stretch, and a spurious break/hold
  during that gappy period can produce a stale-looking state right at the
  boundary where reliable data begins. Not something to fix in the
  scanner; a reason to distrust results from any run whose warmup crosses
  the gappy boundary.

- **Deferred to Phase 2+ (not built)**: QuantHarness-derived Analysis
  layer (TradeIdea, NO_TRADE as a legal outcome), ActiveThesis lifecycle,
  EntryTrigger vocabulary, execution. Full design in the approved
  architecture plan (see conversation/plan history — not persisted as a
  repo file, only in `RESEARCH_NOTES.md` and the code itself).

## Phase 2 — QuantHarness + local Qwen analysis pilot (2026-09-27)

Built the full Phase 2 pipeline on top of the frozen v1 scanner:
`opportunity_snapshot.py` (new episode-centered snapshot, no forward-looking
field exists on the type at all), `trend_analysis.py` (deterministic
trendline/channel fit, adapted from QuantHarness's `graph_util.py` but
taking pre-truncated bars instead of internally slicing a full dataframe),
`chart_renderer.py` (point-in-time chart with a structural
timestamp-after-cutoff assert, not just caller discipline), `vision_analysis.py`
+ `prompts/vision_v1.py` (chart + Python's own trend numbers as text,
vision asked about pattern/structure only, never asked to re-derive slope),
`phase2_analysis.py` + `prompts/phase2_decision_v1.py` (final LONG/SHORT/
NO_TRADE synthesis, `decision` schema-enum-constrained so NO_TRADE is
structurally guaranteed reachable, not just prompted for). `llm_provider.py`
gained `complete_structured`/`complete_vision_structured` (json_schema mode
— confirmed the only reliable structured-output path against this LM
Studio build, see Step 0 note below) plus `seed`/`max_retries`/`vision`
config and per-call latency/retry tracking. `experiment_store.py` gained an
`analyses` table, fully separate from any outcome data.

- **Step 0 (LM Studio `qwen/qwen3-vl-4b` @ `127.0.0.1:1234/v1`) capability
  results, confirmed before building anything on top of them**: text
  completion works; bare `response_format={"type":"json_object"}` is
  **rejected outright** (`400: 'response_format.type' must be 'json_schema'
  or 'text'`) — `json_schema` mode (grammar-constrained) is required and
  confirmed reliable; vision (`image_url` data-URI) works and produces
  chart-accurate descriptions, not generic filler.

- **Real bug found and fixed while testing, not a scanner issue**: the
  QuantHarness-adapted slope optimizer (`_optimize_slope` in
  `trend_analysis.py`) raised on near-perfectly-linear price data — a
  floating-point-noise pivot lands away from either endpoint, and
  perturbing the slope in *either* direction from that pivot instantly
  violates the support/resistance constraint, which the original
  QuantHarness code treated as fatal (`raise Exception`). This is a latent
  robustness gap in the original algorithm, not something specific to this
  port — a real dead/no-volume stretch in market data could hit it too.
  Fixed to fall back to the unoptimized least-squares-anchored line
  (already valid) instead of crashing, so one degenerate episode can't take
  down a whole pilot/batch run.

- **Pilot selection**: `select_pilot_episodes.py` walks the 85 frozen
  episodes chronologically, greedily selecting one whenever it contributes
  a (session, category) pair not yet covered — never reads
  `forward_excursion_research_only`. First attempt used a 15-episode
  "plateau" early-stop and only found 8/16 real pairs before giving up
  mid-dataset; removed the plateau stop (kept the 14-episode cap as the
  only real bound) and it correctly found all **16 distinct (session,
  category) pairs across 12 episodes** — the frozen 85 only ever produce
  16 combinations total (no `(london, volatility_expansion)` or
  `(london, vwap_extension)` episode exists in this window), so 12 was the
  actual ceiling for "one example of everything," not an arbitrary
  shortfall from the 10-15 target.

- **Pilot result (12/12 episodes, all persisted to `data/experiments.db`'s
  `analyses` table)**: every single episode returned **NO_TRADE**
  (confidence 0.60-0.75). NO_TRADE is the correct, intended default for
  ambiguous evidence, not a bug — but a 12/12 rate on a set deliberately
  chosen for category *diversity* (not setup quality) is worth the user's
  attention before deciding what Phase 3 looks like: worth checking whether
  the model is systematically over-conservative, whether the prompt's
  framing nudges toward NO_TRADE, or whether these particular episodes
  (session/category-diverse by construction) just don't happen to contain
  strong setups. Not something to tune from in-place — any prompt change
  is a new `_v2` module per the versioning rule above.
- **Minor output-quality wrinkle, not a correctness issue**: the vision
  model twice returned the JSON string `"null"` for `pattern` instead of a
  true JSON `null`, and left `supporting_factors`/`opposing_factors` empty
  on two decisions despite a substantive `analysis_summary`. Schema-valid
  either way (types matched), just an LLM output-quality quirk on this
  small local model — noted for a future prompt version, not fixed by
  editing `_v1` in place.

- **Per the approved plan, stopped here.** No further episodes beyond the
  12, no TradeIdea/ActiveThesis/entry triggers/execution, no scanner or
  prompt tuning from these results. Next step is the user's call after
  reviewing the pilot.

## Phase 3 — persistent thesis architecture implemented (2026-09-27)

Implemented the full approved Phase 3 design (MarketAnalysis, ActiveThesis,
CANDIDATE→ARMED TradePlan lifecycle, six-setup vocabulary incl.
`MOMENTUM_CONTINUATION`, the two-clock replay driver) as new, additive
files/tables. Nothing in Phase 1, Phase 2 V1 (`analyses`,
`diagnostic_forced_direction`), or Phase 2 V2 (`analyses_v2`, stopped at
26/73) was modified — verified row counts unchanged after all schema
additions (12/12/26 respectively).

New files: `bar_conditions.py`, `thesis_config.py`, `trade_plan_config.py`,
`mtf_4h_candle_context.py`, `market_analysis.py`, `setup_definitions.py`,
`trade_plan_candidates.py`, `active_thesis.py`, `thesis_llm.py`,
`post_entry_review.py`, `trade_plan.py`, `trade_plan_arming.py`,
`trade_plan_replay.py`, `prompts/thesis_formation_v1.py`,
`prompts/thesis_review_v1.py`, `prompts/trade_plan_v1.py`,
`prompts/post_entry_review_v1.py`, `test_no_lookahead_phase3.py`,
`run_phase3.py`. Extended (additive only): `output_quality.py`
(`check_anchoring_suspected`, `check_thesis_output_quality`),
`experiment_store.py` (6 new tables: `market_analyses`, `active_theses`,
`trade_plans`, `trade_plan_arming_events`, `trade_plan_outcomes`,
`post_entry_reviews`).

- **Test results**: 12/12 offline unit/no-lookahead tests pass, plus a
  live round-trip through all 4 new dynamically-built schemas against LM
  Studio (`qwen/qwen3-vl-4b`) — 13/13 total.
- **Bugs found and fixed during implementation, not after a failed run**:
  (1) two column/placeholder-count mismatches in the new `active_theses`/
  `post_entry_reviews` INSERT statements (same class of bug as the earlier
  `analyses`/`diagnostic_forced_direction` mistakes — now systematically
  verified via a schema-vs-placeholder count check before running
  anything); (2) a genuine correctness bug in `thesis_llm.py`'s
  `run_thesis_review`: `parsed.get("updated_supporting_evidence", []) or
  thesis.supporting_evidence` silently fell back to the PRIOR version's
  (stale) evidence whenever the model legitimately reported an empty list
  — meaning "all supporting evidence has evaporated" would have been
  silently overwritten with old evidence instead of correctly persisted as
  empty. Fixed to use the model's array directly (these fields are
  schema-required, always present). Same class of bug fixed for
  `opposing_evidence`/`confirmation_still_needed`.
- **Implementation-level decisions not spelled out in the approved
  design, made explicit here rather than silently**: (a) on `REVERSE`, the
  new opposite-direction thesis is formed via a fresh `ThesisFormation`
  call against the same `MarketAnalysis` (reusing existing code) rather
  than trying to construct it from `ThesisReview`'s sparse
  `reversal_direction` field alone; (b) `PULLBACK_HOLD`'s candidate
  structure is restricted to the trendline reference only (`TrendlineSupport`/
  `TrendlineResistance`), not extended to arbitrary trend-aligned static
  levels, for v1 simplicity; (c) `invalidation_conditions` entries require
  an explicit `adverse_direction` field so the forced-invalidation circuit
  breaker (`active_thesis.check_forced_invalidation`) has enough structure
  to evaluate them deterministically.
- **`run_phase3.py` (the two-clock driver) is written but NOT executed** —
  syntax/import-checked only. Its per-bar orchestration (state transitions,
  deferred next-bar-open fills, plan/thesis persistence sequencing) has not
  been exercised end-to-end against real data; that is exactly what the
  upcoming mechanical smoke test is for, and real bugs should be expected
  to surface there.
- **Stopped here per explicit instruction** — no historical replay run,
  not even the 1-2 day smoke test, until separately approved.

## Phase 2 pilot — NO_TRADE diagnosis (2026-09-27)

Built a diagnostic-only forced-direction pass (`prompts/forced_direction_v1.py`
+ `phase2_diagnostic.py` + `run_phase2_diagnostic.py`, persisted to a
separate `diagnostic_forced_direction` table, never touching `analyses`)
that reuses each pilot episode's exact already-computed snapshot/trend/
vision and asks a different question — forced LONG-vs-SHORT, NO_TRADE
unavailable — to separate "the model sees a lean but abstains" from "the
model genuinely sees balanced evidence." Full episode-by-episode audit in
`audit_phase2_pilot.py` / `data/phase2_pilot_audit.json`.

- **The forced-direction call is well-grounded, not random**: its chosen
  direction matched `TrendAnalysis.channel_direction` in all 9/9 episodes
  where that deterministic trend was rising or falling, and its own stated
  `directional_confidence` was 0.75 in every one of those 9 and exactly
  0.65 in all 3 episodes where the deterministic trend was flat. The model
  is tracking Python's own numbers, not guessing.
- **`MODEL_OUTPUT_QUALITY` issues in 7/12 episodes** — the single biggest
  category, and worth fixing before drawing conclusions about the model's
  reasoning quality: `supporting_factors`/`opposing_factors` left empty
  despite a substantive `analysis_summary` (2 episodes); vision returning
  the JSON string `"null"` instead of true null for `pattern` (2 episodes,
  same wrinkle noted in the first pilot writeup); a factual error reading
  `broken_holding` on a High level as bearish when it's structurally a
  bullish continuation signal; and one episode (`685c4a881498dea0`) where
  the diagnostic's own bull_case/bear_case contents didn't logically
  support the direction it chose — that one episode's diagnostic reading
  should be treated as noise, not signal.
- **`CONFLICTING_DIRECTIONAL_EVIDENCE` in 5/12** — mostly a recurring
  15m-bullish-vs-4h-bearish (or the mirror) split on 2025-09-22, i.e. real
  evidence, not a model artifact.
- **The clearest "abstains despite a real lean" case**: episode
  `2dcbd02e79f0a3d6` (2025-11-05 00:15, asia) had the cleanest
  multi-timeframe-aligned bearish read in the whole pilot (15m/60m/4h all
  bearish, 4h histogram at its most extreme value in the pilot, price
  retraced to a favorable short-the-bounce location) and the most
  confident/coherent diagnostic response (SHORT 0.75, explicit "downtrend
  confirmed by both support and resistance slopes") — yet the real
  decision was still NO_TRADE 0.75 with a fairly generic stated reason.
  Also the most textbook-bullish *location* in the pilot
  (`90fe7b18ee276a07`, price essentially at trend-channel support with a
  confirmed MACD bottom reversal) was NO_TRADE at only 0.60 confidence.
- **Not yet touched**: no scanner change, no prompt edit, no forward/
  outcome data read anywhere in this diagnosis. Any prompt fix is a new
  `_v2` module per the versioning rule, pending the user's direction.

## Phase 2 — frozen-prediction outcome evaluation (2026-09-27)

**Methodology, stated up front so this is never mistaken for a backtest
later**: the original Phase 2 decision (all 12 NO_TRADE) and the diagnostic
forced-direction call (7 LONG / 5 SHORT) were both generated and persisted
*before* any forward/outcome data was read for this experiment —
`evaluate_frozen_directional_outcomes.py` only reads already-committed
`analyses`/`diagnostic_forced_direction` rows plus the raw 1-minute bar
file, and writes a separate `data/phase2_frozen_outcomes.json`; nothing was
merged back into `OpportunitySnapshot`. No entry/stop/target/sizing exists,
so this measures directional correlation only, not a strategy result —
"win rate"/"profit"/"Sharpe"/"expectancy" do not apply and aren't reported.
Cross-checked two episodes' MFE/MAE against the already-existing
direction-agnostic `forward_excursion_research_only` figures from the
original episode review — exact match, confirming the fresh computation is
correct.

- **The frozen LONG/SHORT reads are directionally noisy and dominated by
  outliers, not a clean edge.** LONG (n=7) showed positive median net
  change at all three horizons (30/60/120m: +28.5/+20.0/+50.5pts). SHORT
  (n=5) showed **negative** median net change at all three horizons
  (-4.75/-13.0/-42.5pts) — only 1 of 5 SHORT calls (episode
  `2dcbd02e79f0a3d6`) ended up net-favorable, and it was a large win
  (+180.75pts at 120m) that's carrying the whole SHORT side.
- **Counter-intuitive result: the pre-declared STRONG/ALIGNED subgroup
  (episodes 3, 9, 11, 12 — identified before any outcome was read, for
  having the cleanest multi-timeframe-aligned evidence) did *not*
  outperform the CONFLICTING/RANGE subgroup (episodes 2, 4, 5, 6, 7, 8) —
  if anything the reverse at longer horizons**: strong/aligned median net
  change was -47.5pts (60m) / -84.0pts (120m) vs. conflicting/range's
  +41.25pts (60m) / +62.75pts (120m). This is driven by two large adverse
  moves inside the "strong" group: episode `87023210fce06e9f` (SHORT, the
  95th-percentile-volatility episode already flagged pre-outcome as having
  an oversold-bounce tension) saw a 244pt adverse move against its SHORT
  thesis by 120m, and episode `b26bc793fe5bd62b` (SHORT) saw a 186pt
  adverse move. Both were already flagged pre-outcome as having *some*
  countervailing evidence (oversold RSI risk on the former, a large gap to
  support on the latter) — worth noting, but with only 4 episodes in this
  subgroup this is descriptive, not a statistically meaningful comparison.
- **Bottom line for now: the frozen directional read does not show a clean
  or reliable signal in this 12-episode sample** — real positive results
  exist (the LONG side, several individual big winners) but they're mixed
  with equally large losses on the SHORT side and in the "cleanest
  evidence" subgroup specifically. Not proof of no signal either — n=12 on
  effectively 2 trading days (2025-09-22 and 2025-11-05) is far too small
  and non-independent (7 of 12 episodes are the same morning) to conclude
  anything either way. Do not read this as either "the model works" or
  "the model doesn't work."

**Engineering diagnosis (4 issues from the blind audit, not yet fixed):**

1. **Stale/far-away levels persist because `broken_holding` has no
   distance-based exit at all** (`session_tracker.py`'s `_LevelWatch.
   update()`) — confirmed real bug, not a data artifact for this one:
   `lost`/`reclaimed` states DO reset to `far` once `distance > approach_
   dist` (40 ticks/10pts default), but `broken_holding` only transitions
   onward when price crosses back through the level to the pre-break side
   — if price just keeps running away in the breakout direction, it stays
   `broken_holding` forever, no matter the distance (confirmed:
   episode `8faa25d3e99f435a`'s `PDL broken_holding (+222.5pts)`, episode
   `87023210fce06e9f`'s `AsiaLow broken_holding (-289.25pts)`). Proposed
   fix (not applied): add a relevance filter at the Phase 2 snapshot/prompt-
   building layer (not in frozen `session_tracker.py`) that excludes a
   level from prompt text when `abs(distance_pts)` exceeds some multiple of
   `daily_atr` or `approach_dist`, while keeping the full raw level list in
   the persisted snapshot for auditability. Keeps Phase 1 untouched.
2. **Episode 10's PDH/AsiaHigh "missed" confluence is intended threshold
   behavior, not a bug.** PDH=24747.50, AsiaHigh=24750.00 — 2.50pts apart.
   `confluence_tolerance_ticks=8` × `tick_size=0.25` = 2.0pt tolerance.
   2.50 > 2.0, so it's correctly excluded by the current (frozen v1) number
   — a genuine near-miss under the exact configured threshold, not an
   implementation or eligibility bug. Worth a note for a future v2 tuning
   discussion, not a fix now.
3. **Vision's `"null"` string vs JSON `null`** is a model generation choice
   the schema doesn't prevent — a nullable string field (`["string",
   "null"]`) permits both the literal `null` and the string content
   `"null"`; grammar constraint enforces type-validity, not which valid
   value the model picks. Proposed fix (not applied): a semantics-preserving
   normalization in the parser (`vision_analysis.py`) — if a nullable
   string field's value case-insensitively equals `"null"`, treat it as
   `None`. No prompt or model change needed.
## Phase 2 V2 — directional/tradeability split + blinded 73-episode run (2026-09-27)

Per the user's direction (not optimizing against the 12 revealed outcomes,
not moving to TradeIdea), built V2 on top of the same frozen Phase 1
scanner and unchanged snapshot/trend/chart/vision pipeline: one call now
returns two independent, always-answered fields — `directional_view`
(LONG|SHORT, forced, no abstain — replaces the separate forced-direction
diagnostic call entirely) and `tradeability` (TRADE|NO_TRADE, a genuinely
separate judgment, with `tradeability_reason` always populated either way).
New files: `level_relevance.py`, `output_quality.py`,
`prompts/phase2_decision_v2.py` (`PROMPT_VERSION="v2"`,
`prompts/phase2_decision_v1.py` untouched), `phase2_analysis_v2.py`,
`select_remaining_episodes.py`, `run_phase2_v2.py`,
`test_no_lookahead_phase2_v2.py` (3/3 passing incl. a live schema round-
trip). `experiment_store.py` gained an additive `analyses_v2` table — the
V1 pilot's `analyses`/`diagnostic_forced_direction` rows are untouched.

- **Level relevance fix, principled and NOT tuned to the 12 outcomes**:
  a level is shown to the model only if `state != "far"` and
  `abs(distance_pts) <= 1.0 x daily_atr` — daily_atr is already this
  codebase's own volatility yardstick (used for vwap_extension/
  volatility_expansion), so "further than a typical day's range" is a
  structural, interpretable cutoff. Explicitly NOT chosen to erase the two
  previously-flagged stale examples: at ~250pt daily_atr, `+222.5pts`
  stays inside the cutoff (still shown) while `-289.25pts` falls outside
  (dropped) — that asymmetry was left alone on purpose. Filters both the
  structured levels block AND matching scanner-reasoning lines in lockstep
  (`level_relevance.relevant_levels`/`relevant_reasoning`), applied only
  at the V2 prompt-construction layer — `OpportunitySnapshot`'s raw,
  persisted level list is untouched, and `analyses_v2` separately persists
  `levels_shown_to_model_json` for full reproducibility.
- **"null"-string and empty-array fixes are annotate-and-normalize, not
  silent**: `output_quality.validate_and_normalize_v2` converts a
  case-insensitive `"null"` string to real `None` (recording
  `NULL_STRING_NORMALIZED:<field>`) and flags (never fabricates content
  for) empty factor arrays alongside a substantive summary, a trivial
  `tradeability_reason`, or a TRADE call paired with near-coin-flip
  directional confidence and no factors (aimed at the V1 pilot's one
  internally-incoherent episode). Flags are diagnostic only — never fed
  back into the prompt within a run.
- **Smoke test (3 episodes) confirmed the pipeline end-to-end** before the
  full run: schema parsed cleanly every time, `tradeability_reason` was
  substantive and correctly explained "clear directional lean, but
  NO_TRADE because X" in all 3 — exactly the capability this experiment
  was designed to unlock. One quality flag still fired twice
  (`EMPTY_FACTOR_ARRAY...:directional_bull_case`, all-SHORT episodes with
  no bullish counter-evidence offered) — noted, not iterated on mid-run
  per the frozen-prompt protocol.
- **Episode selection**: all 73 frozen-v1 episodes not already in the V1
  pilot (`select_remaining_episodes.py`), zero coverage/diversity
  heuristic, zero overlap confirmed programmatically with the pilot's 12.
- Full 73-episode run launched; outcome evaluation (directional MFE/MAE/
  net-change, and the core new TRADE-vs-NO_TRADE comparison) is explicitly
  deferred to a separate step after all 73 predictions are frozen and
  persisted, per the approved plan.

4. **Empty `supporting_factors`/`opposing_factors` despite full prose** is
   primarily a small-model multi-field-consistency limitation, not a
   schema/parsing/provider bug — the array is structurally valid (schema
   guarantees a well-formed array, not a non-empty or summary-consistent
   one); the 4B model sometimes doesn't distribute the same reasoning it
   used in `analysis_summary` into the parallel array fields within one
   generation. Proposed fixes (not applied, would require a new `_v2`
   prompt): reorder schema/prompt fields so the arrays are requested before
   `analysis_summary` (front-loaded fields tend to get more consistent
   effort); add an explicit instruction that factors must reflect the same
   evidence discussed in the summary; and/or persist a quality-flag when
   arrays are empty but the summary is non-trivial, so this is visible
   without a fresh audit each time.

## LLM decision-layer project (started 2026-09-26)

Architecture proposal approved and Milestone 1 implemented: a swappable
TAKE/SKIP decision layer sits between `backtest_manual_week.py`'s
deterministic cross/fail candidate detection and trade execution, so a rule
(`RuleBasedStrategy`, reproducing the original `require_macd_turn` gate
exactly) or a local LLM (`LLMStrategy`, via `llm_provider.py` against
LM Studio's OpenAI-compatible endpoint) can be swapped in without the
backtest engine caring which one produced the decision. New modules:
`market_snapshot.py` (point-in-time context bundle, no-lookahead enforced by
construction — see `test_no_lookahead.py`), `decision_strategy.py`
(`Decision`/`Outcome`/`Strategy`/`RuleBasedStrategy`/`LLMStrategy`),
`llm_provider.py`, `prompts/take_skip_v1.py`, `experiment_store.py` (SQLite,
one row per candidate per experiment, TAKE or SKIP alike), `run_experiment.py`
(the A-vs-B CLI). Full design in `/Users/germainmac/.claude/plans/velvety-knitting-nest.md`.

- **Verified the refactor doesn't change baseline behavior.** Re-ran the
  Sep 14 MYM week (previously: 6 trades, 4W/2L, +$48.25, PF 4.11) through
  `RuleBasedStrategy` post-refactor — identical result, bit-for-bit. Safe to
  trust `run_experiment.py`'s Experiment A as the same baseline documented
  elsewhere in this file.

- **The MACD gate is doing real filtering work, not just noise reduction.**
  Same week: 427 total cross/fail candidates, only 6 passed the gate (TAKE).
  The 421 it SKIPped would have collectively LOST -$1,011.75 if all taken
  anyway (forward-simulated), versus the 6 taken netting +$48.25. First
  concrete evidence the deterministic filter has real value, not just noise
  reduction — sets a real bar for `LLMStrategy` to clear in Experiment B.

- **Candidate volume is high — 427 in a 4-day week.** `LLMStrategy` hasn't
  been run yet (needs LM Studio serving Qwen3-VL-4B locally, not yet
  confirmed running) but at ~427 candidates/week, a full apples-to-apples
  comparison means ~427 local LLM calls per week tested. Worth watching
  whether this is practical time-wise once Experiment B actually runs — may
  need candidate-volume reduction (e.g. only the TAKE-eligible ones, or a
  cheaper pre-filter) if it's too slow, but don't preemptively narrow the
  candidate stream before seeing real numbers.

- **Fail-closed on LLM errors, by design.** `prompts/take_skip_v1.py`'s
  `parse_response` defaults to SKIP on any malformed/unparseable LLM output
  (never TAKE) — mirrors the "SKIP is the safe default" principle from
  QuantHarness's review (where forcing a directional call on every
  invocation, never HOLD, was flagged as a real design flaw). Also:
  `llm_provider.py`'s `OpenAICompatProvider.complete()` raises a clear
  `ConnectionError` on an unreachable LM Studio server rather than a raw
  httpx traceback, and `run_experiment.py` catches it to abort that
  experiment leg cleanly (on the first failed candidate, not after
  burning through all 427) instead of crashing the whole run or writing
  hundreds of bogus "connection failed" rows into `experiment_store.py`.

## Strategy findings

- **(SUPERSEDED twice — see below) Early MYM entry-logic history.** First
  pass used MYM's tick size wrong (see instrument-spec bug below), giving
  16/22/18 trades and inflated PnL across the three test weeks. After fixing
  specs but before the MACD gate, immediate-entry-on-cross/fail alone gave:
  Jul 27: 16 trades +$31.00 PF 1.29; Aug 24: 22 trades -$55.00 PF 0.53;
  Sep 14: 18 trades +$29.75 PF 1.34 (net +$5.75, roughly breakeven, but
  16-22 trades/week is way above the 3-4/week cadence originally wanted).
  Superseded again by the MACD-gated version below.

- **Current entry logic (2026-09-19+): cross/fail + 15m MACD "substantial
  reversal" gate, no retest requirement.** `backtest_manual_week.py` now
  enters immediately on a cross/fail (no break-and-retest step — that was
  tried and dropped, see mechanics section) but ONLY if the 15m MACD line
  showed a genuine rounding turn recently: extreme vs. its own 100-bar
  range (top/bottom 20th percentile), a 5-bar-smoothed curve that persists
  2+ bars, checked over the trailing ~2 hours (`mtf_indicators.macd_line_shape`
  + `15m_macd_bottom_recent`/`15m_macd_top_recent`). Three-week MYM result:
  Jul 27: 6 trades, 3W/3L, +$27.00, PF 1.53.
  Aug 24: 7 trades, 2W/5L, -$41.00, PF 0.34.
  Sep 14: 6 trades, 4W/2L, +$48.25, PF 4.11.
  Net: 19 trades, 9W/10L, +$34.25 — roughly breakeven, carried entirely by
  Sep 14. Aug 24 lost because one level (L4/53716) again dominated with 5 of
  7 trades that week (same single-hyperactive-level pattern as the pre-cooldown
  whipsaw, just less extreme). Cross vs. fail across all three weeks: cross
  7 trades (4W/3L, -$2.50), fail 12 trades (5W/7L, +$36.75) — still no
  consistent winner, consistent with every earlier check.

- **Shorting into an uptrend loses; buying dips works.** On the NQ weekly
  swing-extreme fade (`backtest_weekly_swing_fade.py`), longs (buying weekly
  lows) were profitable (+$3,432.50) while shorts (fading weekly highs) lost
  (-$3,750.00) over the same 14-week sample — because NQ spent that whole
  window trending up. Adding a 20-day SMA trend filter (skip shorts when
  price > SMA) flipped the strategy from -$317.50 to +$2,832.50, mostly by
  suppressing shorts down to 6 trades. Caveat: this sample never included a
  downtrend, so we haven't verified the filter helps (rather than just
  "happens to agree with the only regime we tested").

- **Risk-per-trade must be capped, or backtest PnL is fake-good.**
  Uncapped stop distance on the swing-extreme fade let a single bad week
  (NQ, Oct 13-16 2025, three shorts into a strong rally) lose $12,630 in one
  week, driving a $21,492.50 max drawdown against only $15,477.50 total
  profit. Capping risk to $500/contract (tightening the stop, not skipping
  the trade) cut max drawdown to $2,807.50 but also erased most of the
  profit (-$317.50) — the edge was living almost entirely in a few
  outsized, undersized-stop trending trades. Always check whether a
  backtest's profit depends on a handful of oversized winners before
  trusting the headline PnL.

- **One-outlier-trade risk.** The NQ weekly floor-pivot fade
  (`backtest_weekly_pivots.py`) showed 6.7% win rate but net +$3,006.67 —
  entirely because 1 of 15 trades won +$11,336.67 (a genuine move, not a
  bug, but still just one trade). Any time win rate is very low but PnL is
  positive, check whether the result is one lucky trade before believing
  there's an edge.

## Execution / mechanics gotchas (found through building, not trading)

- **Break-and-retest entries were tried and dropped for the manual-level
  system.** Mirrored `trading_agent/atr_state_machine.py`'s proven
  break->retest->entry pattern (don't enter on the first cross/fail, wait
  for price to pull back and hold before entering). On MYM week of Jul 27,
  this cut trades 16->13 and made PnL worse (+$31.00 -> -$66.25) — some
  retests held for one bar then failed anyway (retest and invalidation
  distances were both 6 ticks, too close together to mean much). Removed
  in favor of immediate entry gated by the MACD "substantial reversal"
  condition instead (see strategy findings above). Might be worth
  revisiting with wider retest/invalidation separation if the MACD gate
  alone doesn't hold up.

- **Two-condition-AND gates need slack in the timing, or they never fire.**
  First attempt at the MACD gate required the MACD condition on the EXACT
  same 1-minute bar as the retest/signal — cut 13 trades to 1, then to 0
  when tightened further. Root cause (confirmed by tracing an actual
  example): price often resolves a cross/fail within 1-15 minutes of first
  approaching a level, leaving almost no window for an independent MACD
  condition to coincide. Fix: check whether the MACD condition occurred
  ANYWHERE in the trailing ~2 hours (`_recent` rolling-window columns in
  `mtf_indicators.py`), not on the literal same bar. General lesson: when
  gating one event on another independent event, decouple the timing with
  a real lookback window rather than requiring exact coincidence.

- **Levels need a cooldown after a trade closes, or they whipsaw.** Without
  one, `backtest_manual_week.py` generated 10 trades at a single level (MYM
  L1) inside a 50-minute window as price chopped around it, mostly losses.
  A 4-hour per-level cooldown after a trade closes fixed this (14 trades →
  3 trades on the same week/levels, PnL -$92.50 → +$912.50). Default is now
  `cooldown_hours=4.0` in `Config`.

- **Session window matters a lot — most signals happen outside NY hours.**
  Restricting `backtest_manual_week.py` to 8am-4pm ET missed 8 of 11 trades
  in a single MYM week test; opening it to the full ~23hr Globex session
  (00:00-23:59 ET) picked up otherwise-missed touches on every level,
  including one (L3) that never got tested at all during NY hours. Default
  is now full-session in `Config`.

- **A symbol missing from `INSTRUMENT_SPECS` fails silently, not loudly.**
  `backtest_weekly_swing_fade.py`'s `INSTRUMENT_SPECS` dict (shared by
  `backtest_manual_week.py`) had no entry for MYM, so `.get(symbol, ...NQ)`
  quietly fell back to NQ's tick size (0.25) and dollar value ($20/point)
  instead of MYM's real ones (tick size 1.0, $0.50/point — the "$0.50" was
  right there in the Topstep contract description the whole time). This
  silently invalidated three weeks of MYM backtest results (see above) —
  both the PnL (40x overstated) and the actual trades taken (tick-based
  zone widths were 4x too narrow) were wrong. Lesson: when adding a new
  symbol, add it to `INSTRUMENT_SPECS` explicitly and don't rely on the
  NQ fallback — it won't error, it'll just quietly produce wrong numbers.

- **Contract roll months aren't universally quarterly.** MES/MNQ/M2K/MYM
  roll quarterly (Mar/Jun/Sep/Dec, month codes H/M/U/Z), but Gold (GC/MGC)
  rolls bimonthly (Feb/Apr/Jun/Aug/Oct/Dec). `data_downloader.py`'s
  continuous-download logic originally assumed quarterly for everything,
  which computed a nonexistent MGC.U26 (September) contract and crashed the
  download. Fixed via `_SYMBOL_ROLL_MONTHS` per-symbol override.

## Data quality

- **`data/NQZ5_1m.csv` is gappy.** Only late Sept - mid Dec 2025 has full
  1-minute density (~6,900 bars/week). Everything from late Dec 2024 through
  mid-September 2025 has a handful to a few hundred bars/week and should not
  be used for anything that depends on accurate weekly H/L/C (pivots,
  session ranges, etc.). `weekly_levels.py` flags this automatically
  (`reliable` field / `*` marker) — check it before trusting a week's levels
  on any file, not just NQZ5.
- ES/GC/CL single-contract CSVs (`data/ESZ5_1m.csv`, `GCZ5_1m.csv`,
  `CLZ5_1m.csv`) haven't been checked for the same gap issue yet.
- Topstep's history API only goes back so far per contract — MNQ, MGC, and
  MYM continuous pulls all came back starting mid-to-late in the requested
  range (e.g. MNQ requested from 2026-05-01, actual data starts 2026-06-24)
  even though the download command didn't error.

## Open questions to revisit

- Does the trend-filter finding (skip shorts above SMA) hold up once we
  have a real downtrend in the sample?
- Does the fade > breakout finding hold up past 2 weeks of MYM data?
- Which specific indicator tags (from `mtf_indicators.py`) actually
  correlate with win rate once there's enough trades logged to matter
  (need dozens of trades per tag, not 2-3)?
- One level dominating a week's trade count keeps recurring (MYM 53716 on
  Aug 24 — 9 touches pre-MACD-gate, 5 of 7 trades post-gate) and both times
  that was the losing week. Worth tracking: is a single level firing this
  often itself a bad sign (choppy, not real S/R that week), independent of
  whatever else the setup requires?
