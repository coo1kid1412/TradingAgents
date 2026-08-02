"""Recorded-snapshot benchmark for the deterministic A-share fast path."""

from __future__ import annotations

import gc
import json
import math
import sqlite3
import tempfile
import time
import tracemalloc
from datetime import datetime, timedelta
from pathlib import Path
from unittest import TestCase, main
from zoneinfo import ZoneInfo

from tradingagents.harness.market_warning.adapters.feishu_notifier import FeishuNotifier
from tradingagents.harness.market_warning.adapters.sqlite_repository import (
    SQLiteWarningRepository,
)
from tradingagents.harness.market_warning.domain import (
    DataStatus,
    Evidence,
    FeatureSnapshot,
    Market,
    MarketDataPoint,
    RawMarketSnapshot,
)
from tradingagents.harness.market_warning.rule_policy import (
    evaluate_a_share_rules,
    load_rule_manifest,
)
from tradingagents.harness.market_warning.rule_service import RuleMarketWarningService
from tradingagents.harness.market_warning.runner import EvaluationSlot, FastScanCoordinator


SHANGHAI = ZoneInfo("Asia/Shanghai")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = Path(__file__).with_name("rule_manifest_v1.json")
BENCHMARK_PATH = (
    PROJECT_ROOT / "harness_data/models/market_warning/rule-v1-benchmark.json"
)
RUNS = 100


class RecordedDataPort:
    def load_snapshot(
        self, market: Market, as_of_time: datetime, session_slot: str
    ) -> RawMarketSnapshot:
        values = {
            "index_price": 3500.0,
            "index_change_pct": -0.10,
            "open": 3505.0,
            "high": 3510.0,
            "low": 3490.0,
            "close": 3500.0,
            "breadth_up_pct": 60.0,
            "breadth_above_ma20_pct": 60.0,
            "volume": 1_000_000.0,
        }
        points = tuple(
            MarketDataPoint(
                market=market,
                symbol="000001.SH",
                field=field,
                value=value,
                data_time=as_of_time,
                fetched_at=as_of_time,
                source="recorded_rt_k",
                quality_status=DataStatus.FRESH,
            )
            for field, value in values.items()
        )
        return RawMarketSnapshot(
            market=market,
            as_of_time=as_of_time,
            session_slot=session_slot,
            points=points,
            data_status=DataStatus.FRESH,
            source_times={"recorded_rt_k": as_of_time},
        )


class RecordedFeatureStrategy:
    _FEATURES = {
        "return_1d": -0.001,
        "return_5d": 0.02,
        "return_60d": 0.20,
        "drawdown_20d": -0.02,
        "ma20_distance": 0.01,
        "volatility_ratio_5d_20d": 1.60,
        "range_zscore_20d": 1.0,
        "close_location": 0.5,
        "audited_ohlc_return_1d": -0.001,
        "breadth_up_pct": 60.0,
        "breadth_above_ma20_pct": 60.0,
        "industry_decline_pct": 40.0,
        "limit_down_pct": 0.0,
        "realtime_breadth_coverage_pct": 100.0,
        "realtime_breadth_staleness_minutes": 1.0,
        "margin_balance_growth_20d": 0.0,
        "margin_balance_contracting_from_high": False,
        "turnover_percentile_20d": 0.5,
        "market_phase": "FIRST_SHOCK",
    }

    def build(self, raw: RawMarketSnapshot, _history) -> FeatureSnapshot:
        evidence = tuple(
            Evidence(
                evidence_id=f"recorded:{name}:{raw.as_of_time.isoformat()}",
                group="feature",
                summary=f"Recorded benchmark feature {name}.",
                value=value,
                source="recorded_fixture",
                as_of_time=raw.as_of_time,
            )
            for name, value in self._FEATURES.items()
        )
        return FeatureSnapshot(
            market=raw.market,
            as_of_time=raw.as_of_time,
            session_slot=raw.session_slot,
            feature_version="market-warning-v2",
            features=dict(self._FEATURES),
            evidence=evidence,
            data_quality=DataStatus.FRESH,
            reliability_grade="A",
            source_times={"recorded_rt_k": raw.as_of_time},
        )


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1))
    return ordered[index]


class RuleFastPathBenchmarkTests(TestCase):
    def test_one_hundred_recorded_scans_meet_runtime_memory_and_idempotency_gates(
        self,
    ) -> None:
        manifest = load_rule_manifest(MANIFEST_PATH)
        sent_messages: list[str] = []
        base_time = datetime(2026, 8, 3, 9, 35, tzinfo=SHANGHAI)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "benchmark.db"
            repository = SQLiteWarningRepository(db_path)
            notifier = FeishuNotifier(
                repository,
                sender=lambda message: sent_messages.append(message),
            )
            service = RuleMarketWarningService(
                data_port=RecordedDataPort(),
                feature_strategy=RecordedFeatureStrategy(),
                rule_evaluator=lambda snapshot, previous_assessment=None: (
                    evaluate_a_share_rules(snapshot, manifest, previous_assessment)
                ),
                repository=repository,
                notifier=notifier,
                report_root=root / "reports",
                engine_version=manifest.engine_version,
                manifest_sha256=manifest.manifest_sha256,
            )
            coordinator = FastScanCoordinator(repository, mode="rule_v1")

            first = service.evaluate(Market.A_SHARE, base_time, "intraday-0935")
            duplicate = service.evaluate(Market.A_SHARE, base_time, "intraday-0935")
            self.assertEqual(first.decision.final_level.value, "ORANGE")
            self.assertEqual(duplicate.decision_id, first.decision_id)
            self.assertEqual(len(sent_messages), 1)
            first_screen = sent_messages[0].split("## 触发证据", 1)[0]
            self.assertTrue(first_screen.startswith("# 【橙灯：提前防守】"))
            self.assertIn("立即操作", first_screen)
            self.assertIn("入场门", first_screen)
            self.assertIn("数据截至", first_screen)
            self.assertIn("规则分数不是概率", first_screen)

            durations: list[float] = []
            current_memory: list[int] = []
            tracemalloc.start()
            try:
                for index in range(1, RUNS + 1):
                    as_of = base_time + timedelta(minutes=10 * index)
                    slot = EvaluationSlot(
                        market=Market.A_SHARE,
                        as_of_time=as_of,
                        session_slot=f"intraday-recorded-{index:03d}",
                        local_trade_date=as_of.date(),
                    )
                    started = time.perf_counter()
                    result = coordinator.execute(
                        slot,
                        lambda at=as_of, name=slot.session_slot: service.evaluate(
                            Market.A_SHARE, at, name
                        ),
                    )
                    durations.append(time.perf_counter() - started)
                    self.assertIsNone(result.error_class)
                    gc.collect()
                    current_memory.append(tracemalloc.get_traced_memory()[0])
            finally:
                _, peak_memory = tracemalloc.get_traced_memory()
                tracemalloc.stop()

            with sqlite3.connect(db_path) as connection:
                run_rows = connection.execute(
                    "SELECT llm_calls FROM market_warning_runs ORDER BY id"
                ).fetchall()
                sent_rows = connection.execute(
                    "SELECT COUNT(*) FROM market_warning_alerts WHERE push_status = 'sent'"
                ).fetchone()[0]

        monotonic_growth = all(
            later >= earlier
            for earlier, later in zip(current_memory, current_memory[1:])
        )
        memory_growth_bytes = current_memory[-1] - current_memory[0]
        summary = {
            "ready": True,
            "engine_version": manifest.engine_version,
            "manifest_sha256": manifest.manifest_sha256,
            "runs": RUNS,
            "p50_seconds": round(_percentile(durations, 0.50), 6),
            "p95_seconds": round(_percentile(durations, 0.95), 6),
            "max_seconds": round(max(durations), 6),
            "llm_calls": sum(row[0] for row in run_rows),
            "sent_alerts": int(sent_rows),
            "duplicate_alerts": max(0, len(sent_messages) - int(sent_rows)),
            "memory_monotonic_growth": monotonic_growth,
            "memory_growth_bytes": memory_growth_bytes,
            "peak_memory_bytes": peak_memory,
        }

        self.assertEqual(len(run_rows), RUNS)
        self.assertEqual(summary["llm_calls"], 0)
        self.assertEqual(summary["sent_alerts"], 1)
        self.assertEqual(summary["duplicate_alerts"], 0)
        self.assertLess(summary["p95_seconds"], 30.0)
        self.assertFalse(monotonic_growth)
        self.assertLess(memory_growth_bytes, 2 * 1024 * 1024)

        BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
        BENCHMARK_PATH.write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
