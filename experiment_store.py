#!/usr/bin/env python3
"""
Reproducible experiment storage — SQLite, following the same plain-sqlite3 /
CREATE TABLE IF NOT EXISTS style as trading_agent/storage.py.

One `experiments` row per run (a symbol/date-range/strategy/model
combination), one `decisions` row per candidate setup evaluated within that
run — TAKE or SKIP alike, `outcome_type` distinguishing a real fill
("actual") from a forward-simulated what-if ("hypothetical"). No vector/
embedding table — deliberately out of scope until similarity-retrieval
memory is a later, separate experiment.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from active_thesis import ActiveThesis
from decision_strategy import Decision, Outcome
from market_analysis import MarketAnalysis
from market_snapshot import MarketSnapshot
from opportunity_scanner import Opportunity
from opportunity_snapshot import OpportunitySnapshot
from phase2_analysis import AnalysisResult
from phase2_analysis_v2 import AnalysisResultV2
from phase2_diagnostic import ForcedDirectionResult
from post_entry_review import PostEntryReviewResult
from trade_plan import TradePlan
from trade_plan_arming import ArmingEvent
from trend_analysis import TrendAnalysis
from vision_analysis import VisionAnalysis

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id         TEXT PRIMARY KEY,
    created_at            TEXT,
    label                 TEXT,
    symbol                TEXT,
    date_range_start      TEXT,
    date_range_end        TEXT,
    strategy_class        TEXT,
    strategy_config_json  TEXT,
    model_name            TEXT,
    model_version         TEXT,
    prompt_version        TEXT,
    inference_params_json TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id             TEXT PRIMARY KEY,
    experiment_id           TEXT REFERENCES experiments(experiment_id),
    timestamp               TEXT,
    symbol                  TEXT,
    level_name              TEXT,
    setup_type              TEXT,
    setup_direction         TEXT,
    snapshot_json           TEXT,
    raw_llm_response        TEXT,
    action                  TEXT,
    confidence              REAL,
    thesis                  TEXT,
    supporting_factors_json TEXT,
    opposing_factors_json   TEXT,
    invalidation            TEXT,
    entry_px  REAL, stop_px REAL, target_px REAL,
    exit_time TEXT, exit_px REAL, exit_reason TEXT,
    pnl_usd REAL, pnl_r REAL, mfe REAL, mae REAL,
    outcome_type            TEXT
);

CREATE TABLE IF NOT EXISTS scan_runs (
    scan_run_id        TEXT PRIMARY KEY,
    created_at         TEXT,
    symbol             TEXT,
    date_range_start   TEXT,
    date_range_end     TEXT,
    session_config_json  TEXT,
    scanner_config_json  TEXT
);

CREATE TABLE IF NOT EXISTS opportunities (
    opportunity_id     TEXT PRIMARY KEY,
    scan_run_id        TEXT,
    timestamp          TEXT,
    symbol             TEXT,
    sessions_json       TEXT,
    is_opportunity     INTEGER,
    categories_json    TEXT,
    reasoning_json     TEXT,
    level_states_json  TEXT,
    confluence_json    TEXT,
    episode_id         TEXT,
    is_new_episode     INTEGER
);

CREATE TABLE IF NOT EXISTS analyses (
    analysis_id               TEXT PRIMARY KEY,
    episode_id                TEXT,
    timestamp                 TEXT,
    symbol                    TEXT,
    snapshot_json             TEXT,
    trend_analysis_json       TEXT,
    chart_path                TEXT,
    vision_analysis_json      TEXT,
    final_prompt              TEXT,
    raw_response              TEXT,
    decision                  TEXT,
    confidence                REAL,
    setup                     TEXT,
    thesis                    TEXT,
    reason                    TEXT,
    supporting_factors_json   TEXT,
    opposing_factors_json     TEXT,
    key_levels_json           TEXT,
    invalidation_concept      TEXT,
    analysis_summary          TEXT,
    model_name                TEXT,
    provider                  TEXT,
    prompt_version            TEXT,
    inference_settings_json   TEXT,
    latency_s                 REAL,
    retry_count               INTEGER,
    parse_errors              TEXT,
    created_at                TEXT
);

-- Phase 3 tables (additive only -- everything above is untouched) --------

CREATE TABLE IF NOT EXISTS market_analyses (
    market_analysis_id   TEXT PRIMARY KEY,
    timestamp             TEXT,
    symbol                TEXT,
    snapshot_json         TEXT,
    trend_analysis_json   TEXT,
    vision_analysis_json  TEXT,
    candle_4h_json        TEXT,
    chart_path            TEXT,
    created_at            TEXT
);

CREATE TABLE IF NOT EXISTS active_theses (
    row_id                          TEXT PRIMARY KEY,
    thesis_id                       TEXT,
    version                         INTEGER,
    parent_thesis_id                TEXT,
    created_at                      TEXT,
    last_reviewed_at                TEXT,
    symbol                          TEXT,
    direction                       TEXT,
    state                           TEXT,
    thesis_statement                TEXT,
    intended_horizon_bars           INTEGER,
    confidence                      REAL,
    supporting_evidence_json        TEXT,
    opposing_evidence_json          TEXT,
    structural_basis_json           TEXT,
    important_levels_json           TEXT,
    expected_market_behavior        TEXT,
    confirmation_still_needed_json  TEXT,
    invalidation_conditions_json    TEXT,
    structural_objectives_json      TEXT,
    risks_json                      TEXT,
    evidence_strengthened_json      TEXT,
    evidence_weakened_json          TEXT,
    evidence_disappeared_json       TEXT,
    new_evidence_against_json       TEXT,
    original_assumptions_still_hold INTEGER,
    opposite_thesis_evidence_strength REAL,
    what_changed_since_previous     TEXT,
    review_action                   TEXT,
    terminal_reason                 TEXT,
    market_analysis_id              TEXT,
    prior_thesis_version_id         TEXT,
    quality_flags_json              TEXT,
    model_name                      TEXT,
    provider                        TEXT,
    prompt_version                  TEXT,
    inference_settings_json         TEXT,
    raw_response                    TEXT,
    latency_s                       REAL,
    created_at_row                  TEXT
);

CREATE TABLE IF NOT EXISTS trade_plans (
    plan_id                    TEXT PRIMARY KEY,
    thesis_id                  TEXT,
    thesis_version_id_formed   TEXT,
    market_analysis_id         TEXT,
    created_at                 TEXT,
    symbol                     TEXT,
    direction                  TEXT,
    plan_status                TEXT,
    no_plan_reason             TEXT,
    setup_type                 TEXT,
    reference_price            REAL,
    trigger_level_name         TEXT,
    trigger_level_price        REAL,
    trigger_condition          TEXT,
    invalidation_level_name    TEXT,
    invalidation_level_price   REAL,
    invalidation_condition     TEXT,
    target_choice_type         TEXT,
    target_level_name          TEXT,
    target_r_multiple          REAL,
    target_price_hint          REAL,
    max_wait_bars              INTEGER,
    max_hold_bars              INTEGER,
    thesis_restatement         TEXT,
    candidate_levels_json      TEXT,
    model_name                 TEXT,
    provider                   TEXT,
    prompt_version             TEXT,
    inference_settings_json    TEXT,
    raw_response                TEXT,
    latency_s                  REAL,
    created_at_row              TEXT
);

CREATE TABLE IF NOT EXISTS trade_plan_arming_events (
    event_id                      TEXT PRIMARY KEY,
    plan_id                       TEXT,
    timestamp                     TEXT,
    event                         TEXT,
    thesis_version_id_at_event    TEXT,
    created_at                    TEXT
);

CREATE TABLE IF NOT EXISTS trade_plan_outcomes (
    outcome_id           TEXT PRIMARY KEY,
    plan_id               TEXT,
    final_state           TEXT,
    trigger_time          TEXT,
    entry_time            TEXT,
    entry_price           REAL,
    exit_time             TEXT,
    exit_price            REAL,
    exit_reason           TEXT,
    bars_waited           INTEGER,
    bars_held             INTEGER,
    r_multiple_realized   REAL,
    created_at            TEXT
);

CREATE TABLE IF NOT EXISTS post_entry_reviews (
    review_id                          TEXT PRIMARY KEY,
    plan_id                            TEXT,
    timestamp                          TEXT,
    thesis_status                      TEXT,
    evidence_strengthened_json         TEXT,
    evidence_weakened_json             TEXT,
    evidence_disappeared_json          TEXT,
    new_evidence_against_json          TEXT,
    original_assumptions_still_hold    INTEGER,
    opposite_thesis_evidence_strength  REAL,
    reasoning                          TEXT,
    model_name                         TEXT,
    provider                           TEXT,
    prompt_version                     TEXT,
    inference_settings_json            TEXT,
    raw_response                       TEXT,
    latency_s                          REAL,
    created_at                         TEXT
);

CREATE TABLE IF NOT EXISTS diagnostic_forced_direction (
    diagnostic_id             TEXT PRIMARY KEY,
    episode_id                TEXT,
    timestamp                 TEXT,
    symbol                    TEXT,
    snapshot_json             TEXT,
    trend_analysis_json       TEXT,
    vision_analysis_json      TEXT,
    final_prompt              TEXT,
    raw_response              TEXT,
    forced_direction          TEXT,
    directional_confidence    REAL,
    bull_case_json            TEXT,
    bear_case_json            TEXT,
    decisive_factors_json     TEXT,
    model_name                TEXT,
    provider                  TEXT,
    prompt_version            TEXT,
    inference_settings_json   TEXT,
    latency_s                 REAL,
    retry_count               INTEGER,
    parse_errors              TEXT,
    created_at                TEXT
);

CREATE TABLE IF NOT EXISTS analyses_v2 (
    analysis_id                          TEXT PRIMARY KEY,
    episode_id                           TEXT,
    timestamp                            TEXT,
    symbol                                TEXT,
    snapshot_json                        TEXT,
    trend_analysis_json                  TEXT,
    chart_path                           TEXT,
    vision_analysis_json                 TEXT,
    levels_shown_to_model_json           TEXT,
    final_prompt                         TEXT,
    raw_response                         TEXT,
    directional_view                     TEXT,
    directional_confidence               REAL,
    directional_bull_case_json           TEXT,
    directional_bear_case_json           TEXT,
    directional_decisive_factors_json    TEXT,
    tradeability                         TEXT,
    tradeability_confidence              REAL,
    tradeability_reason                  TEXT,
    tradeability_supporting_factors_json TEXT,
    tradeability_opposing_factors_json   TEXT,
    setup                                TEXT,
    key_levels_json                      TEXT,
    invalidation_concept                 TEXT,
    analysis_summary                     TEXT,
    quality_flags_json                   TEXT,
    model_name                           TEXT,
    provider                             TEXT,
    prompt_version                       TEXT,
    inference_settings_json              TEXT,
    latency_s                            REAL,
    retry_count                          INTEGER,
    parse_errors                         TEXT,
    created_at                           TEXT
);
"""


class ExperimentStore:
    def __init__(self, db_path: str | Path = "data/experiments.db"):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def create_experiment(
        self,
        *,
        label: str,
        symbol: str,
        date_range_start,
        date_range_end,
        strategy_class: str,
        strategy_config: dict,
        model_name: str = "",
        model_version: str = "",
        prompt_version: str = "",
        inference_params: Optional[dict] = None,
    ) -> str:
        experiment_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                experiment_id,
                datetime.now(timezone.utc).isoformat(),
                label,
                symbol,
                str(date_range_start),
                str(date_range_end),
                strategy_class,
                json.dumps(strategy_config),
                model_name,
                model_version,
                prompt_version,
                json.dumps(inference_params or {}),
            ),
        )
        self._conn.commit()
        return experiment_id

    def record_decision(
        self,
        *,
        experiment_id: str,
        snapshot: MarketSnapshot,
        decision: Decision,
        outcome: Outcome,
    ) -> str:
        decision_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                decision_id,
                experiment_id,
                snapshot.timestamp.isoformat(),
                snapshot.symbol,
                snapshot.level_name,
                snapshot.setup_type,
                snapshot.setup_direction,
                json.dumps(snapshot.to_dict()),
                decision.raw_response,
                decision.action,
                decision.confidence,
                decision.thesis,
                json.dumps(decision.supporting_factors),
                json.dumps(decision.opposing_factors),
                decision.invalidation,
                outcome.entry_px,
                outcome.stop_px,
                outcome.target_px,
                outcome.exit_time.isoformat() if outcome.exit_time else None,
                outcome.exit_px,
                outcome.exit_reason,
                outcome.pnl_usd,
                outcome.pnl_r,
                outcome.mfe,
                outcome.mae,
                outcome.outcome_type,
            ),
        )
        self._conn.commit()
        return decision_id

    def create_scan_run(
        self,
        *,
        symbol: str,
        date_range_start,
        date_range_end,
        session_config: dict,
        scanner_config: dict,
    ) -> str:
        scan_run_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO scan_runs VALUES (?,?,?,?,?,?,?)",
            (
                scan_run_id,
                datetime.now(timezone.utc).isoformat(),
                symbol,
                str(date_range_start),
                str(date_range_end),
                json.dumps(session_config),
                json.dumps(scanner_config),
            ),
        )
        self._conn.commit()
        return scan_run_id

    def record_opportunity(self, *, scan_run_id: str, opp: Opportunity) -> str:
        opportunity_id = str(uuid.uuid4())
        level_states = [dataclasses.asdict(l) for l in opp.level_states]
        zones = [dataclasses.asdict(z) for z in opp.confluence_zones]
        self._conn.execute(
            "INSERT INTO opportunities VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                opportunity_id,
                scan_run_id,
                opp.timestamp.isoformat(),
                opp.symbol,
                json.dumps(sorted(opp.sessions)),
                int(opp.is_opportunity),
                json.dumps(list(opp.categories)),
                json.dumps(list(opp.reasoning)),
                json.dumps(level_states),
                json.dumps(zones),
                opp.episode_id,
                int(opp.is_new_episode),
            ),
        )
        self._conn.commit()
        return opportunity_id

    def record_analysis(
        self,
        *,
        snapshot: OpportunitySnapshot,
        trend: Optional[TrendAnalysis],
        chart_path: str,
        vision: VisionAnalysis,
        final_prompt: str,
        result: AnalysisResult,
        model_name: str,
        provider: str,
        prompt_version: str,
        inference_settings: dict,
        retry_count: int = 0,
        parse_errors: Optional[str] = None,
    ) -> str:
        analysis_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO analyses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                analysis_id,
                snapshot.episode_id,
                snapshot.timestamp.isoformat(),
                snapshot.symbol,
                json.dumps(snapshot.to_dict()),
                json.dumps(dataclasses.asdict(trend)) if trend is not None else None,
                chart_path,
                json.dumps(dataclasses.asdict(vision)),
                final_prompt,
                result.raw_response,
                result.decision,
                result.confidence,
                result.setup,
                result.thesis,
                result.reason,
                json.dumps(list(result.supporting_factors)),
                json.dumps(list(result.opposing_factors)),
                json.dumps(list(result.key_levels)),
                result.invalidation_concept,
                result.analysis_summary,
                model_name,
                provider,
                prompt_version,
                json.dumps(inference_settings),
                result.latency_s,
                retry_count,
                parse_errors,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return analysis_id

    def record_analysis_v2(
        self,
        *,
        snapshot: OpportunitySnapshot,
        trend: Optional[TrendAnalysis],
        chart_path: str,
        vision: VisionAnalysis,
        levels_shown_to_model: tuple[str, ...],
        final_prompt: str,
        result: AnalysisResultV2,
        model_name: str,
        provider: str,
        prompt_version: str,
        inference_settings: dict,
        retry_count: int = 0,
        parse_errors: Optional[str] = None,
    ) -> str:
        analysis_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO analyses_v2 VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                analysis_id,
                snapshot.episode_id,
                snapshot.timestamp.isoformat(),
                snapshot.symbol,
                json.dumps(snapshot.to_dict()),
                json.dumps(dataclasses.asdict(trend)) if trend is not None else None,
                chart_path,
                json.dumps(dataclasses.asdict(vision)),
                json.dumps(list(levels_shown_to_model)),
                final_prompt,
                result.raw_response,
                result.directional_view,
                result.directional_confidence,
                json.dumps(list(result.directional_bull_case)),
                json.dumps(list(result.directional_bear_case)),
                json.dumps(list(result.directional_decisive_factors)),
                result.tradeability,
                result.tradeability_confidence,
                result.tradeability_reason,
                json.dumps(list(result.tradeability_supporting_factors)),
                json.dumps(list(result.tradeability_opposing_factors)),
                result.setup,
                json.dumps(list(result.key_levels)),
                result.invalidation_concept,
                result.analysis_summary,
                json.dumps(list(result.quality_flags)),
                model_name,
                provider,
                prompt_version,
                json.dumps(inference_settings),
                result.latency_s,
                retry_count,
                parse_errors,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return analysis_id

    def record_market_analysis(self, ma: MarketAnalysis) -> str:
        self._conn.execute(
            "INSERT INTO market_analyses VALUES (?,?,?,?,?,?,?,?,?)",
            (
                ma.market_analysis_id, ma.timestamp.isoformat(), ma.symbol,
                json.dumps(ma.snapshot.to_dict()),
                json.dumps(dataclasses.asdict(ma.trend)) if ma.trend is not None else None,
                json.dumps(dataclasses.asdict(ma.vision)) if ma.vision is not None else None,
                json.dumps({"current": ma.candle_4h_current, "completed": ma.candle_4h_completed}),
                ma.chart_path,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return ma.market_analysis_id

    def record_active_thesis(self, thesis: ActiveThesis, quality_flags: tuple[str, ...] = ()) -> str:
        row_id = f"{thesis.thesis_id}:{thesis.version}"
        self._conn.execute(
            "INSERT INTO active_theses VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                row_id, thesis.thesis_id, thesis.version, thesis.parent_thesis_id,
                thesis.created_at.isoformat(), thesis.last_reviewed_at.isoformat(),
                thesis.symbol, thesis.direction, thesis.state, thesis.thesis_statement,
                thesis.intended_horizon_bars, thesis.confidence,
                json.dumps(list(thesis.supporting_evidence)), json.dumps(list(thesis.opposing_evidence)),
                json.dumps(list(thesis.structural_basis)), json.dumps(list(thesis.important_levels)),
                thesis.expected_market_behavior, json.dumps(list(thesis.confirmation_still_needed)),
                json.dumps(list(thesis.invalidation_conditions)), json.dumps(list(thesis.structural_objectives)),
                json.dumps(list(thesis.risks)),
                json.dumps(list(thesis.evidence_strengthened)), json.dumps(list(thesis.evidence_weakened)),
                json.dumps(list(thesis.evidence_disappeared)), json.dumps(list(thesis.new_evidence_against)),
                int(thesis.original_assumptions_still_hold), thesis.opposite_thesis_evidence_strength,
                thesis.what_changed_since_previous, thesis.review_action, thesis.terminal_reason,
                thesis.market_analysis_id, thesis.prior_thesis_version_id,
                json.dumps(list(quality_flags)),
                thesis.model_name, thesis.provider, thesis.prompt_version,
                json.dumps(thesis.inference_settings_json), thesis.raw_response, thesis.latency_s,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return row_id

    def record_trade_plan(self, plan: TradePlan) -> str:
        self._conn.execute(
            "INSERT INTO trade_plans VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                plan.plan_id, plan.thesis_id, plan.thesis_version_id_formed, plan.market_analysis_id,
                plan.created_at.isoformat(), plan.symbol, plan.direction,
                plan.plan_status, plan.no_plan_reason, plan.setup_type, plan.reference_price,
                plan.trigger_level_name, plan.trigger_level_price, plan.trigger_condition,
                plan.invalidation_level_name, plan.invalidation_level_price, plan.invalidation_condition,
                plan.target_choice_type, plan.target_level_name, plan.target_r_multiple, plan.target_price_hint,
                plan.max_wait_bars, plan.max_hold_bars, plan.thesis_restatement,
                json.dumps(list(plan.candidate_levels_json)),
                plan.model_name, plan.provider, plan.prompt_version,
                json.dumps(plan.inference_settings_json), plan.raw_response, plan.latency_s,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return plan.plan_id

    def record_arming_event(self, event: ArmingEvent) -> str:
        event_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO trade_plan_arming_events VALUES (?,?,?,?,?,?)",
            (event_id, event.plan_id, event.timestamp.isoformat(), event.event,
             event.thesis_version_id_at_event, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()
        return event_id

    def record_trade_plan_outcome(
        self, *, plan_id: str, final_state: str,
        trigger_time: Optional[datetime] = None, entry_time: Optional[datetime] = None,
        entry_price: Optional[float] = None, exit_time: Optional[datetime] = None,
        exit_price: Optional[float] = None, exit_reason: Optional[str] = None,
        bars_waited: Optional[int] = None, bars_held: Optional[int] = None,
        r_multiple_realized: Optional[float] = None,
    ) -> str:
        outcome_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO trade_plan_outcomes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                outcome_id, plan_id, final_state,
                trigger_time.isoformat() if trigger_time else None,
                entry_time.isoformat() if entry_time else None, entry_price,
                exit_time.isoformat() if exit_time else None, exit_price, exit_reason,
                bars_waited, bars_held, r_multiple_realized,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return outcome_id

    def record_post_entry_review(
        self, *, plan_id: str, timestamp: datetime, result: PostEntryReviewResult,
        model_name: str, provider: str, prompt_version: str, inference_settings: dict,
    ) -> str:
        review_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO post_entry_reviews VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                review_id, plan_id, timestamp.isoformat(), result.thesis_status,
                json.dumps(list(result.evidence_strengthened)), json.dumps(list(result.evidence_weakened)),
                json.dumps(list(result.evidence_disappeared)), json.dumps(list(result.new_evidence_against)),
                int(result.original_assumptions_still_hold), result.opposite_thesis_evidence_strength,
                result.reasoning, model_name, provider, prompt_version,
                json.dumps(inference_settings), result.raw_response, result.latency_s,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return review_id

    def record_diagnostic_forced_direction(
        self,
        *,
        episode_id: str,
        timestamp: str,
        symbol: str,
        snapshot: dict,
        trend: Optional[dict],
        vision: dict,
        final_prompt: str,
        result: ForcedDirectionResult,
        model_name: str,
        provider: str,
        prompt_version: str,
        inference_settings: dict,
        retry_count: int = 0,
        parse_errors: Optional[str] = None,
    ) -> str:
        diagnostic_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO diagnostic_forced_direction VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                diagnostic_id,
                episode_id,
                timestamp,
                symbol,
                json.dumps(snapshot),
                json.dumps(trend) if trend is not None else None,
                json.dumps(vision),
                final_prompt,
                result.raw_response,
                result.forced_direction,
                result.directional_confidence,
                json.dumps(list(result.bull_case)),
                json.dumps(list(result.bear_case)),
                json.dumps(list(result.decisive_factors)),
                model_name,
                provider,
                prompt_version,
                json.dumps(inference_settings),
                result.latency_s,
                retry_count,
                parse_errors,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return diagnostic_id

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ExperimentStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
