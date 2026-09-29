#!/usr/bin/env python3
"""
Phase 3 historical replay driver -- the "two clocks" (approved plan §2/§13):
a 1-minute clock does deterministic TradePlan monitoring (no LLM), a
15-minute clock drives every LLM call (ThesisFormation/ThesisReview/
TradePlanFormation/PostEntryReview). Reuses the frozen Phase 1 scanner and
the unchanged Phase 2 snapshot/trend/chart/vision pipeline unmodified.

NOT RUN as part of implementation -- per explicit instruction, this file
is written and ready, but the first historical replay (even the 1-2 day
mechanical smoke test) is a separate, later approval step.

Usage (when approved):
    python3 run_phase3.py --file data/NQZ5_1m.csv --symbol NQ \\
        --start 2025-09-22 --end 2025-09-24   # smoke-test window example
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd

from active_thesis import ActiveThesis, check_horizon_expired
from chart_renderer import render
from experiment_store import ExperimentStore
from llm_provider import LLMConfig, OpenAICompatProvider
from market_analysis import build_market_analysis
from mtf_indicators import compute_mtf_features
from opportunity_scanner import compute_scanner_features, scan
from opportunity_snapshot import build_opportunity_snapshot
from post_entry_review import run_post_entry_review
from session_config import SessionConfig
from session_tracker import SCANNER_CONFIG_VERSION, ScannerConfig, SessionTracker
from thesis_config import ThesisConfig
from thesis_llm import run_thesis_formation, run_thesis_review
from trade_plan import TradePlan, form_trade_plan
from trade_plan_arming import ArmingEvent, compute_arming_event, current_plan_status
from trade_plan_config import TradePlanConfig
from trade_plan_replay import (
    Bar,
    PostEntryWatch,
    PreEntryWatch,
    check_post_entry_bar_outcome,
    check_pre_entry_bar_outcome,
)
from trend_analysis import compute_trend_analysis
from vision_analysis import run_vision_analysis
from weekly_levels import build_weekly_bars, compute_pivots

BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "qwen/qwen3-vl-4b"

_TERMINAL_PRE_ENTRY = {"MISSED", "EXPIRED", "INVALIDATED_BEFORE_ENTRY", "ABANDONED_WITH_THESIS"}
_TERMINAL_POST_ENTRY = {"TARGET_HIT", "INVALIDATION_HIT", "THESIS_EXIT", "MAX_HOLD_EXIT"}


class Phase3State:
    """All the mutable state a single-symbol sequential replay needs
    between bars. Deliberately plain (not persisted itself) -- every
    meaningful transition is written to ExperimentStore as it happens."""

    def __init__(self, cfg_thesis: ThesisConfig, cfg_plan: TradePlanConfig, tick_size: float):
        self.cfg_thesis = cfg_thesis
        self.cfg_plan = cfg_plan
        self.tick_size = tick_size

        self.active_thesis: Optional[ActiveThesis] = None
        self.active_plan: Optional[TradePlan] = None
        self.arming_events: list[ArmingEvent] = []
        self.pre_watch: Optional[PreEntryWatch] = None
        self.post_watch: Optional[PostEntryWatch] = None
        self.entry_price: Optional[float] = None
        self.bars_held = 0
        self.thesis_invalidated_flag = False
        self.pending_entry_fill = False  # set when trigger fires; filled at NEXT bar's open
        self.pending_context_events: list[dict] = []  # MISSED/ABANDONED_WITH_THESIS, surfaced to next review


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/NQZ5_1m.csv")
    ap.add_argument("--symbol", default="NQ")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--charts-dir", default="data/phase3_charts")
    ap.add_argument("--db", default="data/experiments.db")
    args = ap.parse_args()

    path = Path(args.file)
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Computing indicators/scanner features (frozen v1 computation, unchanged)...")
    features = compute_mtf_features(df)
    scanner_features = compute_scanner_features(df)
    weekly_pivots = compute_pivots(build_weekly_bars(df))

    session_cfg = SessionConfig()
    scanner_cfg = ScannerConfig()
    assert scanner_cfg.version == SCANNER_CONFIG_VERSION == "v1", "scanner config version drifted from v1"
    thesis_cfg = ThesisConfig()
    plan_cfg = TradePlanConfig()
    tick_size = 0.25

    tracker = SessionTracker(config=session_cfg, scanner_config=scanner_cfg,
                              weekly_pivots=weekly_pivots, tick_size=tick_size)
    provider = OpenAICompatProvider(LLMConfig(base_url=BASE_URL, model=MODEL))
    store = ExperimentStore(db_path=args.db)
    charts_dir = Path(args.charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)
    seen_episodes: set[str] = set()
    st = Phase3State(thesis_cfg, plan_cfg, tick_size)

    print(f"\nReplaying {path} from {args.start} to {args.end} (1-minute clock, 15-minute thesis/plan cadence)...\n")
    for row in df.itertuples():
        tracker.update(row)
        bar = Bar(row.timestamp, row.open, row.high, row.low, row.close)
        in_range = start <= row.timestamp < end

        # ---- deferred next-bar-open entry fill (from a trigger detected on the prior bar) ----
        if in_range and st.pending_entry_fill and st.active_plan is not None:
            st.entry_price = row.open
            st.post_watch = PostEntryWatch(st.active_plan, st.entry_price, plan_cfg, tick_size)
            st.bars_held = 0
            st.pending_entry_fill = False
            print(f"  [{row.timestamp}] ENTERED plan {st.active_plan.plan_id} at {st.entry_price} "
                  f"(setup={st.active_plan.setup_type})")

        # ================= 1-MINUTE CLOCK: deterministic plan monitoring, no LLM =================
        if in_range and st.active_plan is not None and not st.pending_entry_fill:
            macd_hist = features.loc[row.timestamp].get("15m_hist") if row.timestamp in features.index else None
            macd_hist = None if pd.isna(macd_hist) else float(macd_hist) if macd_hist is not None else None

            if st.post_watch is None:
                # CANDIDATE or ARMED -- pre-entry monitoring
                status = current_plan_status(st.arming_events)
                bars_since_created = int((row.timestamp - st.active_plan.created_at).total_seconds() // 60)
                if bars_since_created >= st.active_plan.max_wait_bars:
                    store.record_trade_plan_outcome(plan_id=st.active_plan.plan_id, final_state="EXPIRED",
                                                     bars_waited=bars_since_created)
                    st.pending_context_events.append(
                        {"outcome": "EXPIRED", "setup_type": st.active_plan.setup_type,
                         "timestamp": row.timestamp.isoformat(), "reason": "max_wait_bars elapsed"})
                    st.active_plan, st.pre_watch, st.arming_events = None, None, []
                else:
                    outcome = check_pre_entry_bar_outcome(st.pre_watch, bar, is_armed=(status == "ARMED"),
                                                           macd_15m_hist=macd_hist)
                    if outcome == "ENTERED":
                        st.pending_entry_fill = True
                    elif outcome in ("MISSED", "INVALIDATED_BEFORE_ENTRY"):
                        store.record_trade_plan_outcome(plan_id=st.active_plan.plan_id, final_state=outcome,
                                                         bars_waited=bars_since_created)
                        st.pending_context_events.append(
                            {"outcome": outcome, "setup_type": st.active_plan.setup_type,
                             "timestamp": row.timestamp.isoformat(), "reason": f"{outcome} at bar close"})
                        st.active_plan, st.pre_watch, st.arming_events = None, None, []
            else:
                # ENTERED -- post-entry monitoring; hard boundaries only, PostEntryReview is advisory
                st.bars_held += 1
                outcome = check_post_entry_bar_outcome(
                    st.post_watch, bar, st.bars_held, st.active_plan.max_hold_bars,
                    st.thesis_invalidated_flag, macd_15m_hist=macd_hist,
                )
                if outcome in _TERMINAL_POST_ENTRY:
                    exit_price = (st.post_watch.target_price if outcome == "TARGET_HIT"
                                  else st.active_plan.invalidation_level_price if outcome == "INVALIDATION_HIT"
                                  else row.close)
                    risk = abs(st.entry_price - st.active_plan.invalidation_level_price) \
                        if st.active_plan.invalidation_level_price is not None \
                        else abs(st.entry_price - st.active_plan.reference_price)
                    pnl = (exit_price - st.entry_price) if st.active_plan.direction == "LONG" \
                        else (st.entry_price - exit_price)
                    r_realized = (pnl / risk) if risk else None
                    store.record_trade_plan_outcome(
                        plan_id=st.active_plan.plan_id, final_state=outcome,
                        entry_time=st.active_plan.created_at, entry_price=st.entry_price,
                        exit_time=row.timestamp, exit_price=exit_price, exit_reason=outcome,
                        bars_held=st.bars_held, r_multiple_realized=r_realized,
                    )
                    if st.active_thesis is not None:
                        completed = _terminal_thesis_version(st.active_thesis, "COMPLETED",
                                                              f"TradePlan reached {outcome}")
                        store.record_active_thesis(completed)
                        st.active_thesis = None
                    st.active_plan, st.pre_watch, st.post_watch = None, None, None
                    st.thesis_invalidated_flag = False

        # ================= 15-MINUTE CLOCK: all LLM calls =================
        if in_range and row.timestamp.minute % 15 == 0:
            snapshot = build_opportunity_snapshot(tracker, features, scanner_features,
                                                   scan(tracker=tracker, features=features,
                                                        scanner_features=scanner_features, price=row.close,
                                                        timestamp=row.timestamp, symbol=args.symbol.upper(),
                                                        seen_episodes=seen_episodes, cfg=scanner_cfg),
                                                   df, lookback_bars=60)
            trend = compute_trend_analysis(snapshot.recent_bars, daily_atr=snapshot.daily_atr)

            need_vision = st.active_thesis is not None or snapshot.is_new_episode
            vision, chart_path = None, None
            if need_vision:
                chart_path = str(charts_dir / f"{row.timestamp.strftime('%Y%m%dT%H%M')}.png")
                _, image_b64 = render(snapshot.recent_bars, snapshot_timestamp=row.timestamp,
                                       levels=snapshot.levels, trend=trend, vwap=snapshot.vwap,
                                       out_path=chart_path)
                vision = run_vision_analysis(provider, snapshot, trend, image_b64)

            ma = build_market_analysis(str(uuid.uuid4()), snapshot, trend, df, vision=vision, chart_path=chart_path)
            store.record_market_analysis(ma)

            if st.active_thesis is None:
                if snapshot.is_new_episode:
                    result = run_thesis_formation(provider, ma, thesis_cfg)
                    if result.thesis_status == "CANDIDATE":
                        st.active_thesis = result.thesis
                        store.record_active_thesis(st.active_thesis)
                        print(f"  [{row.timestamp}] NEW THESIS {st.active_thesis.thesis_id[:8]} "
                              f"{st.active_thesis.direction} (CANDIDATE)")
            else:
                if check_horizon_expired(st.active_thesis, row.timestamp):
                    expired = _terminal_thesis_version(st.active_thesis, "EXPIRED", "intended_horizon_bars elapsed")
                    store.record_active_thesis(expired)
                    if st.active_plan is not None:
                        _abandon_active_plan(store, st, row.timestamp, "parent thesis EXPIRED")
                    st.active_thesis = None
                else:
                    review = run_thesis_review(provider, st.active_thesis, ma, tuple(st.pending_context_events))
                    st.pending_context_events = []
                    store.record_active_thesis(review.new_version, quality_flags=review.quality_flags)
                    print(f"  [{row.timestamp}] THESIS {review.new_version.thesis_id[:8]} "
                          f"v{review.new_version.version}: {review.new_version.review_action} "
                          f"-> {review.new_version.state}")

                    if review.new_version.state == "REVERSED":
                        if st.active_plan is not None:
                            _abandon_active_plan(store, st, row.timestamp, "parent thesis REVERSED")
                        st.active_thesis = None
                        reversal_result = run_thesis_formation(provider, ma, thesis_cfg)
                        if reversal_result.thesis_status == "CANDIDATE":
                            import dataclasses as _dc
                            new_thesis = _dc.replace(
                                reversal_result.thesis,
                                parent_thesis_id=review.new_version.thesis_id,
                            )
                            st.active_thesis = new_thesis
                            store.record_active_thesis(new_thesis)
                            print(f"  [{row.timestamp}] REVERSAL -> NEW THESIS {new_thesis.thesis_id[:8]} "
                                  f"{new_thesis.direction} (CANDIDATE)")
                    elif review.new_version.state in ("INVALIDATED",):
                        if st.active_plan is not None:
                            _abandon_active_plan(store, st, row.timestamp, "parent thesis INVALIDATED")
                        st.active_thesis = None
                    else:
                        st.active_thesis = review.new_version

                    # Post-entry review, if a position is open under this thesis.
                    if st.active_thesis is not None and st.post_watch is not None:
                        per = run_post_entry_review(
                            provider, st.active_thesis.thesis_statement, st.active_thesis.direction,
                            st.active_plan.setup_type, ma,
                        )
                        store.record_post_entry_review(
                            plan_id=st.active_plan.plan_id, timestamp=row.timestamp, result=per,
                            model_name=provider.config.model, provider="OpenAICompatProvider",
                            prompt_version="v1", inference_settings={"temperature": provider.config.temperature},
                        )
                        st.thesis_invalidated_flag = (per.thesis_status == "THESIS_INVALIDATED")

                    # Plan formation / arming, if the thesis (still alive) is actionable.
                    if st.active_thesis is not None and st.active_thesis.state in ("DEVELOPING", "STRONG"):
                        if st.active_plan is None:
                            plan = form_trade_plan(provider, st.active_thesis, ma, plan_cfg, tick_size)
                            store.record_trade_plan(plan)
                            if plan.plan_status == "CANDIDATE":
                                st.active_plan = plan
                                st.pre_watch = PreEntryWatch(plan, plan_cfg, tick_size)
                                st.arming_events = []
                        if st.active_plan is not None and st.post_watch is None:
                            status = current_plan_status(st.arming_events)
                            event = compute_arming_event(
                                status, st.active_thesis.state, st.active_plan.plan_id,
                                row.timestamp, f"{st.active_thesis.thesis_id}:{st.active_thesis.version}",
                            )
                            if event is not None:
                                st.arming_events.append(event)
                                store.record_arming_event(event)
                                print(f"  [{row.timestamp}] plan {st.active_plan.plan_id[:8]} {event.event}")

    store.close()
    print("\nReplay complete.")


def _terminal_thesis_version(thesis: ActiveThesis, state: str, reason: str) -> ActiveThesis:
    import dataclasses as _dc
    return _dc.replace(
        thesis, version=thesis.version + 1, state=state, terminal_reason=reason,
        review_action="INVALIDATE" if state == "INVALIDATED" else thesis.review_action,
        prior_thesis_version_id=f"{thesis.thesis_id}:{thesis.version}",
    )


def _abandon_active_plan(store: ExperimentStore, st: Phase3State, timestamp, reason: str) -> None:
    store.record_trade_plan_outcome(plan_id=st.active_plan.plan_id, final_state="ABANDONED_WITH_THESIS",
                                     exit_time=timestamp, exit_reason=reason)
    st.pending_context_events.append({"outcome": "ABANDONED_WITH_THESIS", "setup_type": st.active_plan.setup_type,
                                       "timestamp": timestamp.isoformat(), "reason": reason})
    st.active_plan, st.pre_watch, st.post_watch, st.thesis_invalidated_flag = None, None, None, False


if __name__ == "__main__":
    main()
