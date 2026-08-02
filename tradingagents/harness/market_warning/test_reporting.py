from __future__ import annotations

import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest import TestCase, main
from unittest import mock

from tradingagents.harness.market_warning.domain import (
    DataStatus,
    DecisionSource,
    FeatureSnapshot,
    FinalWarningDecision,
    LLMContextAssessment,
    Market,
    MarketPhase,
    QuantRiskAssessment,
    RiskLevel,
    RuleLayer,
    RuleRiskAssessment,
    RunnerResult,
    TriggeredRule,
)
from tradingagents.harness.market_warning.reporting import (
    render_premarket_report,
    render_upgrade_report,
    write_report,
)


NOW = datetime(2026, 8, 3, 0, 30, tzinfo=timezone.utc)


def _snapshot(
    market: Market = Market.A_SHARE,
    status: DataStatus = DataStatus.FRESH,
) -> FeatureSnapshot:
    return FeatureSnapshot(
        market=market,
        as_of_time=NOW,
        session_slot="premarket",
        feature_version="market-warning-v2",
        features={"market_phase": "FIRST_SHOCK"},
        evidence=(),
        data_quality=status,
        reliability_grade="A" if status == DataStatus.FRESH else "C",
        source_times={"fixture:last": NOW},
    )


def _quant(
    *,
    phase: MarketPhase = MarketPhase.FIRST_SHOCK,
    one_day: float = 0.04,
    three_day: float = 0.08,
    reliability: str = "A",
) -> QuantRiskAssessment:
    return QuantRiskAssessment(
        crash_1d_probability=one_day,
        crash_3d_probability=three_day,
        market_phase=phase,
        base_rate_1d=0.01 if reliability != "UNAVAILABLE" else 0.0,
        base_rate_3d=0.02 if reliability != "UNAVAILABLE" else 0.0,
        reliability_grade=reliability,
        model_version="warning-model-v2" if reliability != "UNAVAILABLE" else "unavailable",
        calibration_version="platt-v2" if reliability != "UNAVAILABLE" else "unavailable",
        top_contributors=(
            {"feature": "breadth_up_pct", "contribution": 1.25},
            {"feature": "vix_change_5d", "contribution": 0.80},
            {"feature": "ma20_distance", "contribution": -0.35},
            {"feature": "ignored_fourth", "contribution": 0.20},
        ),
    )


def _decision(
    level: RiskLevel,
    *,
    transition: str = "UNCHANGED",
    status: DataStatus = DataStatus.FRESH,
) -> FinalWarningDecision:
    actions = {
        RiskLevel.GREEN: ("OPEN", 100.0, "HOLD"),
        RiskLevel.YELLOW: ("OPEN", 100.0, "HOLD"),
        RiskLevel.ORANGE: ("CONDITIONAL", 3.0, "HOLD_OR_REDUCE"),
        RiskLevel.RED: ("WAIT", 0.0, "REDUCE"),
        RiskLevel.UNKNOWN: ("WAIT", 0.0, "HOLD"),
    }
    gate, cap, holding = actions[level]
    return FinalWarningDecision(
        baseline_level=level,
        final_level=level,
        state_transition=transition,
        entry_gate=gate,
        new_position_cap_pct=cap,
        holding_action=holding,
        push_required=level in {RiskLevel.ORANGE, RiskLevel.RED},
        decision_reasons=("calibrated probability, not certainty",),
        data_status=status,
    )


def _context() -> LLMContextAssessment:
    return LLMContextAssessment(
        market_scenario="流动性与市场宽度同步走弱",
        causal_chain=("宽度先行恶化", "波动率随后上升"),
        supporting_evidence_ids=("ev-1", "ev-2"),
        conflicting_evidence_ids=("ev-3",),
        overlooked_risks=("政策对冲可能降低尾部风险",),
        recommended_risk_level=RiskLevel.ORANGE,
        confidence=0.78,
        action_reason="在确认前压低新增风险",
        reasoning_status="validated",
    )


def _result(
    level: RiskLevel,
    *,
    market: Market = Market.A_SHARE,
    phase: MarketPhase = MarketPhase.FIRST_SHOCK,
    status: DataStatus = DataStatus.FRESH,
    transition: str = "UNCHANGED",
    context: LLMContextAssessment | None = None,
    shadow: bool = False,
) -> RunnerResult:
    snapshot_status = DataStatus.SHADOW if shadow else status
    quant = _quant(
        phase=phase,
        reliability="UNAVAILABLE" if level == RiskLevel.UNKNOWN else "A",
        one_day=0.0 if level == RiskLevel.UNKNOWN else 0.04,
        three_day=0.0 if level == RiskLevel.UNKNOWN else 0.08,
    )
    return RunnerResult(
        market=market,
        as_of_time=NOW,
        session_slot="premarket",
        feature_snapshot=_snapshot(market, snapshot_status),
        quant_assessment=quant,
        context_assessment=context,
        decision=_decision(level, transition=transition, status=snapshot_status),
    )


def _rule_result(
    level: RiskLevel = RiskLevel.ORANGE,
    *,
    slot: str = "premarket",
    transition: str = "INITIAL_ORANGE",
) -> RunnerResult:
    snapshot = replace(
        _snapshot(),
        session_slot=slot,
        features={
            "market_phase": "FIRST_SHOCK",
            "realtime_breadth_coverage_pct": 0.98,
            "realtime_breadth_staleness_minutes": 1.5,
        },
    )
    rules = (
        TriggeredRule(
            rule_id="A-PRESSURE-BREADTH",
            layer=RuleLayer.PRESSURE,
            severity_points=2,
            observed_value=-0.43,
            threshold_description="上涨家数占比较20日基线下降至少35个百分点",
            evidence_ids=("breadth-now", "breadth-baseline"),
        ),
        TriggeredRule(
            rule_id="A-PRESSURE-LIMIT-DOWN",
            layer=RuleLayer.PRESSURE,
            severity_points=1,
            observed_value=0.012,
            threshold_description="跌停家数占比不低于1%",
            evidence_ids=("limit-down-now",),
        ),
        TriggeredRule(
            rule_id="A-VULNERABILITY-TREND",
            layer=RuleLayer.VULNERABILITY,
            severity_points=1,
            observed_value=-0.018,
            threshold_description="指数位于20日均线下方",
            evidence_ids=("index-daily",),
        ),
        TriggeredRule(
            rule_id="A-FOURTH-RULE",
            layer=RuleLayer.CONTINUATION,
            severity_points=1,
            observed_value=True,
            threshold_description="第四条证据不应出现在短告警中",
            evidence_ids=("fourth",),
        ),
    )
    assessment = RuleRiskAssessment(
        market=Market.A_SHARE,
        as_of_time=NOW,
        engine_version="rule-v1.0.0",
        manifest_sha256="a" * 64,
        risk_level=level,
        risk_score=5.0,
        market_phase=MarketPhase.FIRST_SHOCK,
        triggered_rules=rules,
        missing_optional_groups=(),
        reliability_grade="A",
        evaluation_latency_ms=18.0,
    )
    decision = replace(
        _decision(level, transition=transition),
        decision_source=DecisionSource.RULE_V1,
    )
    return RunnerResult(
        market=Market.A_SHARE,
        as_of_time=NOW,
        session_slot=slot,
        feature_snapshot=snapshot,
        rule_assessment=assessment,
        decision=decision,
    )


class ReportGoldenTests(TestCase):
    def test_rule_premarket_puts_action_data_and_score_disclaimer_first(self) -> None:
        report = render_premarket_report(_rule_result(), None)
        first_screen = "\n".join(report.splitlines()[:14])

        self.assertTrue(report.startswith("# A股大盘骤跌预警\n\n## 【橙灯：提前防守】"))
        self.assertIn("**立即操作：", first_screen)
        self.assertIn("**入场门：", first_screen)
        self.assertIn("**新增仓位上限：", first_screen)
        self.assertIn("**持仓动作：", first_screen)
        self.assertIn("规则生产版", first_screen)
        self.assertIn("数据截至", first_screen)
        self.assertIn("可靠度：`A`", first_screen)
        self.assertIn("规则分数：`5.0/10`（规则分数不是概率）", first_screen)
        self.assertNotIn("骤跌概率", report)
        self.assertNotIn("0.00%", report)
        self.assertNotIn("<think>", report.lower())

    def test_rule_intraday_alert_is_short_and_shows_only_top_three_rules(self) -> None:
        result = _rule_result(
            RiskLevel.RED,
            slot="intraday-0935",
            transition="UPGRADE_ORANGE_TO_RED",
        )

        report = render_upgrade_report(result, _decision(RiskLevel.ORANGE))

        self.assertTrue(report.startswith("# 【红灯：风险确认】"))
        self.assertIn("ORANGE -> RED", report)
        self.assertIn("**入场门：", report)
        self.assertIn("**新增仓位上限：", report)
        self.assertIn("**持仓动作：", report)
        self.assertIn("数据截至", report)
        self.assertIn("可靠度：`A`", report)
        self.assertIn("规则分数不是概率", report)
        self.assertEqual(report.count("规则 `A-"), 3)
        self.assertNotIn("A-FOURTH-RULE", report)
        self.assertNotIn("骤跌概率", report)

    def test_rule_m3_section_is_explanation_only(self) -> None:
        context = replace(
            _context(),
            recommended_risk_level=RiskLevel.RED,
            confidence=0.99,
            action_reason="this field must not become a recommendation",
        )
        result = replace(_rule_result(), context_assessment=context)

        report = render_premarket_report(result, None)

        self.assertIn("M3 情景解释", report)
        self.assertIn("不改变规则灯号和操作约束", report)
        self.assertNotIn("建议：", report)
        self.assertNotIn("99.00%", report)
        self.assertNotIn("this field must not become a recommendation", report)

    def test_green_premarket_has_action_block_first_and_stable_section_order(self) -> None:
        report = render_premarket_report(_result(RiskLevel.GREEN), None)
        lines = report.splitlines()

        self.assertEqual(lines[0], "> **【绿灯 GREEN】立即操作：可按既定计划参与，但仍须执行个股止损。**")
        self.assertEqual(lines[1], "> 入场门：`OPEN` | 新增仓位上限：`100%` | 持仓动作：`HOLD`")
        ordered = [
            "## 概率判断",
            "## 市场阶段",
            "## 相比上一份",
            "## 主要驱动",
            "## M3 情景校验",
            "## 数据与模型",
        ]
        positions = [report.index(item) for item in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("概率估计，不代表确定会发生", report)
        self.assertEqual(report.count("ignored_fourth"), 0)

    def test_reports_cover_the_six_operating_states(self) -> None:
        previous_orange = _decision(RiskLevel.ORANGE)
        cases = (
            (_result(RiskLevel.GREEN), None, "【绿灯 GREEN】", "FIRST_SHOCK"),
            (_result(RiskLevel.ORANGE, context=_context()), None, "【橙灯 ORANGE】", "首次冲击"),
            (_result(RiskLevel.RED, phase=MarketPhase.CONTINUATION), None, "【红灯 RED】", "延续下跌"),
            (_result(RiskLevel.UNKNOWN, status=DataStatus.STALE), None, "【未知 UNKNOWN】", "数据不足不等于低风险"),
            (_result(RiskLevel.YELLOW, market=Market.US, shadow=True), None, "【黄灯 YELLOW】", "影子运行"),
            (
                _result(RiskLevel.RED, transition="UPGRADE_ORANGE_TO_RED"),
                previous_orange,
                "【红灯 RED】",
                "ORANGE -> RED",
            ),
        )
        for result, previous, lamp, marker in cases:
            with self.subTest(lamp=lamp, marker=marker):
                report = (
                    render_upgrade_report(result, previous)
                    if result.decision and result.decision.state_transition == "UPGRADE_ORANGE_TO_RED"
                    else render_premarket_report(result, previous)
                )
                self.assertTrue(report.startswith("> **"))
                self.assertIn(lamp, report)
                self.assertIn(marker, report)
                self.assertIn("入场门", report.splitlines()[1])
                self.assertIn("新增仓位上限", report.splitlines()[1])

    def test_report_redacts_private_reasoning_errors_and_secrets(self) -> None:
        unsafe = LLMContextAssessment(
            market_scenario="<think>private chain</think> Traceback API error sk-secret",
            causal_chain=("MINIMAX_API_KEY=secret",),
            supporting_evidence_ids=("ev-1", "ev-2"),
            conflicting_evidence_ids=("ev-3",),
            overlooked_risks=("{\"raw\": \"json\"}",),
            recommended_risk_level=RiskLevel.ORANGE,
            confidence=0.8,
            action_reason="provider error: credential secret",
            reasoning_status="validated",
        )

        report = render_premarket_report(_result(RiskLevel.ORANGE, context=unsafe), None)

        for forbidden in ("<think>", "private chain", "Traceback", "API error", "sk-secret", "MINIMAX_API_KEY", "{\"raw\""):
            self.assertNotIn(forbidden, report)
        self.assertIn("[内容已脱敏]", report)

    def test_report_lists_conflicting_evidence_and_source_times(self) -> None:
        report = render_premarket_report(
            _result(RiskLevel.ORANGE, context=_context()), None
        )

        self.assertIn("反向证据ID：`ev-3`", report)
        self.assertIn("fixture:last", report)
        self.assertIn(NOW.isoformat(), report)

    def test_upgrade_report_contains_only_change_evidence_and_action(self) -> None:
        result = _result(
            RiskLevel.RED,
            transition="UPGRADE_ORANGE_TO_RED",
            context=_context(),
        )
        snapshot = replace(
            result.feature_snapshot,
            features={
                **dict(result.feature_snapshot.features),
                "breadth_deterioration_transition": True,
            },
        )
        result = replace(result, feature_snapshot=snapshot, session_slot="intraday-0935")

        report = render_upgrade_report(result, _decision(RiskLevel.ORANGE))

        self.assertIn("## 变化", report)
        self.assertIn("ORANGE -> RED", report)
        self.assertIn("## 触发证据", report)
        self.assertIn("breadth_deterioration_transition", report)
        self.assertNotIn("## 概率判断", report)
        self.assertNotIn("## 数据与模型", report)

    def test_silent_intraday_poll_is_titled_evaluation_not_upgrade(self) -> None:
        result = _result(RiskLevel.GREEN)
        result = replace(
            result,
            session_slot="intraday-0935",
            feature_snapshot=replace(result.feature_snapshot, session_slot="intraday-0935"),
        )

        report = render_upgrade_report(result, _decision(RiskLevel.GREEN))

        self.assertIn("（盘中评估）", report)
        self.assertNotIn("（盘中升级）", report)

    def test_metadata_and_authorization_shaped_text_are_redacted(self) -> None:
        unsafe_context = replace(
            _context(),
            market_scenario="Authorization: Bearer SECRET",
            action_reason="system prompt: Return JSON only",
        )
        result = _result(RiskLevel.ORANGE, context=unsafe_context)
        result = replace(
            result,
            feature_snapshot=replace(
                result.feature_snapshot,
                feature_version="<THINK>private</THINK>",
                reliability_grade="api_key: TOPSECRET",
            ),
            quant_assessment=replace(
                result.quant_assessment,
                model_version="token=TOPSECRET",
            ),
        )

        report = render_premarket_report(result, None)

        for forbidden in (
            "Bearer SECRET",
            "TOPSECRET",
            "Return JSON only",
            "<THINK>",
            "private",
        ):
            self.assertNotIn(forbidden, report)
        self.assertIn("[内容已脱敏]", report)

    def test_write_report_uses_market_local_date_and_stable_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = _result(RiskLevel.GREEN)

            path = write_report(result, None, Path(directory))

            expected = Path(directory) / "a_share" / "2026-08-03" / "0830-premarket.md"
            self.assertEqual(path, expected)
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_text(encoding="utf-8"), render_premarket_report(result, None))
            self.assertEqual(list(expected.parent.glob("*.tmp")), [])

    def test_atomic_replace_failure_preserves_previous_report_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = _result(RiskLevel.GREEN)
            target = Path(directory) / "a_share" / "2026-08-03" / "0830-premarket.md"
            target.parent.mkdir(parents=True)
            target.write_text("previous-report", encoding="utf-8")

            with mock.patch(
                "tradingagents.harness.market_warning.reporting.os.replace",
                side_effect=OSError("replace failed"),
            ), self.assertRaises(OSError):
                write_report(result, None, Path(directory))

            self.assertEqual(target.read_text(encoding="utf-8"), "previous-report")
            self.assertEqual(list(target.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    main()
