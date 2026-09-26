# Backtest Research Notes

Running log of things noticed while backtesting that could clue us into what
actually works. Add to this every time a backtest turns up something
non-obvious — a pattern, a data quirk, a design mistake that skewed results.
Don't log routine run output here, only things worth remembering next time.

---

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
