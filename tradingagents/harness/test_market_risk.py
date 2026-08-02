"""市场风险快照的确定性行为回归测试。

运行：python tradingagents/harness/test_market_risk.py
（项目自跑风格，不依赖 pytest；原 pytest fixture tmp_path/monkeypatch/capsys 已就地手写。）
"""

import contextlib
import datetime as _dt
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from tradingagents.harness.market_risk import (
    compose_effective_market_gate,
    compute_market_risk_snapshot,
    infer_market,
    save_market_risk_snapshot,
    load_latest_market_risk_snapshot,
    load_market_risk_for_ticker,
    load_market_warning_for_ticker,
    enforce_snapshot_freshness,
)
from tradingagents.harness.market_warning.adapters.sqlite_repository import SQLiteWarningRepository
from tradingagents.harness.market_warning.domain import (
    DataStatus,
    DecisionSource,
    FeatureSnapshot,
    FinalWarningDecision,
    Market,
    MarketPhase,
    QuantRiskAssessment,
    RuleRiskAssessment,
)
from tradingagents.harness import market_risk_daily
from tradingagents.harness.market_risk_daily import is_market_trading_day, run_market_risk_daily, _send_feishu_message


def _prices(values):
    return [{"close": value} for value in values]


@contextlib.contextmanager
def _tmp_dir():
    """替代 pytest tmp_path：建临时目录，结束清理。"""
    d = tempfile.mkdtemp(prefix="mktrisk_")
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


@contextlib.contextmanager
def _env(**overrides):
    """替代 monkeypatch.setenv/delenv：覆盖环境变量并在结束时恢复。值为 None 表示删除。"""
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_high_risk_forces_wait_and_caps_position():
    """下跌趋势 + 高波动应禁止追高并压低仓位。"""
    closes = list(range(140, 100, -1))
    snapshot = compute_market_risk_snapshot(
        "a_share", _prices(closes), breadth_pct=25, volatility_pct=35,
    )

    assert snapshot["risk_level"] in ("高", "极高")
    assert snapshot["entry_gate"] == "WAIT"
    assert snapshot["position_cap_pct"] <= 3
    assert snapshot["t_plus_1_bias"] == "偏空"


def test_low_risk_keeps_entry_open():
    """稳定上升趋势且宽度健康时不意外限制个股入场。"""
    closes = [100 + i * 0.4 for i in range(60)]
    snapshot = compute_market_risk_snapshot(
        "us", _prices(closes), breadth_pct=70, volatility_pct=12,
    )

    assert snapshot["risk_level"] == "低"
    assert snapshot["entry_gate"] == "OPEN"
    assert snapshot["position_cap_pct"] == 20
    assert snapshot["t_plus_1_bias"] == "偏多"


def test_missing_price_data_marks_snapshot_unavailable():
    snapshot = compute_market_risk_snapshot("a_share", [], breadth_pct=None, volatility_pct=None)

    assert snapshot["risk_level"] == "数据不足"
    assert snapshot["entry_gate"] == "WAIT"
    assert snapshot["data_status"] == "missing"


def test_snapshot_storage_is_unique_per_market_and_date():
    with _tmp_dir() as tmp_path:
        db_path = tmp_path / "risk.db"
        first = compute_market_risk_snapshot("a_share", _prices([100 + i for i in range(60)]), 70, 10)
        first["as_of_date"] = "2026-06-25"
        first["as_of_time"] = "2026-06-25T08:30:00+08:00"
        save_market_risk_snapshot(first, db_path)

        updated = dict(first, risk_level="中", position_cap_pct=6)
        save_market_risk_snapshot(updated, db_path)
        loaded = load_latest_market_risk_snapshot("a_share", "2026-06-25", db_path)

        assert loaded["risk_level"] == "中"
        assert loaded["position_cap_pct"] == 6
        assert infer_market("NVDA") == "us"
        assert infer_market("300308") == "a_share"
        assert load_market_risk_for_ticker("300308", "2026-06-25", db_path)["market"] == "a_share"


def test_same_day_premarket_snapshot_fails_closed_after_intraday_checkpoint():
    snapshot = compute_market_risk_snapshot(
        "a_share", _prices([100 + i for i in range(60)]), 70, 10,
        as_of_date="2026-07-15", as_of_time="2026-07-15T08:30:00+08:00",
    )
    checked = enforce_snapshot_freshness(
        snapshot, "2026-07-15", analysis_time="2026-07-15T14:18:00+08:00",
    )
    assert checked["data_status"] == "stale"
    assert checked["entry_gate"] == "WAIT"
    assert checked["position_cap_pct"] == 0
    assert checked["required_checkpoint"] == "11:15"


def test_current_intraday_checkpoint_remains_usable():
    snapshot = compute_market_risk_snapshot(
        "a_share", _prices([100 + i for i in range(60)]), 70, 10,
        as_of_date="2026-07-15", as_of_time="2026-07-15T11:16:00+08:00",
    )
    checked = enforce_snapshot_freshness(
        snapshot, "2026-07-15", analysis_time="2026-07-15T14:18:00+08:00",
    )
    assert checked["data_status"] == "fresh"
    assert checked["entry_gate"] == "OPEN"


def test_daily_runner_skips_closed_market_and_does_not_push():
    with _tmp_dir() as tmp_path:
        sent = []
        result = run_market_risk_daily(
            as_of_date="2026-06-20",  # Saturday
            db_path=tmp_path / "risk.db",
            fetch_prices=lambda _ticker, _start, _end: _prices([100 + i for i in range(60)]),
            send_message=sent.append,
        )

        assert result["a_share"]["status"] == "closed"
        assert result["us"]["status"] == "closed"
        assert sent == []
        assert not is_market_trading_day("a_share", "2026-06-20")


def test_daily_runner_persists_and_push_failure_does_not_discard_snapshot():
    with _tmp_dir() as tmp_path:
        def prices(ticker, _start, _end):
            return _prices([100 + i * 0.4 for i in range(60)])

        result = run_market_risk_daily(
            as_of_date="2026-06-25",
            db_path=tmp_path / "risk.db",
            fetch_prices=prices,
            send_message=lambda _text: (_ for _ in ()).throw(RuntimeError("webhook unavailable")),
        )
        stored = load_latest_market_risk_snapshot("a_share", "2026-06-25", tmp_path / "risk.db")

        assert result["a_share"]["status"] == "saved"
        assert result["a_share"]["push_status"] == "failed"
        assert stored is not None and stored["as_of_date"] == "2026-06-25"


def test_daily_runner_can_target_one_market_for_its_local_preopen():
    with _tmp_dir() as tmp_path:
        result = run_market_risk_daily(
            as_of_date="2026-06-25", db_path=tmp_path / "risk.db", markets=("a_share",),
            fetch_prices=lambda _ticker, _start, _end: _prices([100 + i for i in range(60)]),
            send_message=lambda _text: None,
        )

        assert set(result) == {"a_share"}


def test_feishu_sender_uses_open_api_credentials_when_webhook_is_absent():
    """已有飞书应用凭证时不需要额外配置自定义机器人 webhook。"""
    calls = []

    class _Response:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(req, timeout):
        calls.append((req.full_url, json.loads(req.data.decode()), timeout))
        if "tenant_access_token" in req.full_url:
            return _Response({"code": 0, "tenant_access_token": "token"})
        return _Response({"code": 0})

    saved_urlopen = market_risk_daily._ur.urlopen
    with _env(FEISHU_MARKET_RISK_WEBHOOK=None, FEISHU_APP_ID="app-id",
              FEISHU_APP_SECRET="app-secret", FEISHU_USER_OPEN_ID="user-open-id"):
        try:
            market_risk_daily._ur.urlopen = fake_urlopen
            _send_feishu_message("市场风险测试")
        finally:
            market_risk_daily._ur.urlopen = saved_urlopen

    assert len(calls) == 2
    assert calls[0][1] == {"app_id": "app-id", "app_secret": "app-secret"}
    assert calls[1][1]["receive_id"] == "user-open-id"
    assert calls[1][1]["msg_type"] == "text"


def test_cli_treats_dry_run_as_success():
    saved_run = market_risk_daily.run_market_risk_daily
    saved_argv = sys.argv
    buf = io.StringIO()
    try:
        market_risk_daily.run_market_risk_daily = (
            lambda **_kwargs: {"a_share": {"status": "dry_run", "push_status": "skipped"}})
        sys.argv = ["market_risk_daily", "--market", "a_share", "--dry-run"]
        with contextlib.redirect_stdout(buf):
            rc = market_risk_daily.main()
    finally:
        market_risk_daily.run_market_risk_daily = saved_run
        sys.argv = saved_argv
    assert rc == 0
    assert "a_share: dry_run / push=skipped" in buf.getvalue()


def _warning(level="ORANGE", market=Market.A_SHARE, status=DataStatus.FRESH):
    actions = {
        "GREEN": ("OPEN", 100.0),
        "YELLOW": ("OPEN", 100.0),
        "ORANGE": ("CONDITIONAL", 3.0),
        "RED": ("WAIT", 0.0),
        "UNKNOWN": ("WAIT", 0.0),
    }
    gate, cap = actions[level]
    return {
        "market": market.value,
        "as_of_time": "2026-08-03T01:35:00+00:00",
        "session_slot": "intraday-0935",
        "warning_level": level,
        "baseline_level": level,
        "entry_gate": gate,
        "position_cap_pct": cap,
        "holding_action": "REDUCE" if level == "RED" else "HOLD",
        "transition": f"INITIAL_{level}",
        "data_status": status.value,
        "phase": "FIRST_SHOCK",
        "probabilities": {"1d": 0.05, "3d": 0.10},
        "base_rates": {"1d": 0.01, "3d": 0.02},
        "reasons": ["calibrated warning"],
    }


def test_warning_orange_strictly_composes_without_touching_long_term_fields():
    legacy = {
        "market": "a_share", "entry_gate": "OPEN", "position_cap_pct": 20,
        "risk_level": "低", "long_term_rating": "BUY", "target_price_12m": 150,
        "reasons": ["legacy calm"],
    }

    result = compose_effective_market_gate(legacy, _warning("ORANGE"), "a_share")

    assert result["entry_gate"] == "CONDITIONAL"
    assert result["position_cap_pct"] == 3
    assert result["long_term_rating"] == "BUY"
    assert result["target_price_12m"] == 150
    assert result["effective_gate_source"] == "market_warning"
    assert result["warning_level"] == "ORANGE"


def test_warning_red_and_unknown_force_wait_while_legacy_wait_remains_strictest():
    open_legacy = {"market": "a_share", "entry_gate": "OPEN", "position_cap_pct": 20, "reasons": []}
    for level in ("RED", "UNKNOWN"):
        result = compose_effective_market_gate(open_legacy, _warning(level), "a_share")
        assert result["entry_gate"] == "WAIT"
        assert result["position_cap_pct"] == 0

    wait_legacy = {"market": "a_share", "entry_gate": "WAIT", "position_cap_pct": 0, "reasons": []}
    result = compose_effective_market_gate(wait_legacy, _warning("GREEN"), "a_share")
    assert result["entry_gate"] == "WAIT"
    assert result["position_cap_pct"] == 0
    assert result["effective_gate_source"] == "legacy_market_risk"


def test_us_shadow_warning_is_visible_but_does_not_override_production_gate():
    legacy = {"market": "us", "entry_gate": "OPEN", "position_cap_pct": 20, "reasons": []}

    result = compose_effective_market_gate(
        legacy,
        _warning("RED", market=Market.US, status=DataStatus.SHADOW),
        "us",
    )

    assert result["entry_gate"] == "OPEN"
    assert result["position_cap_pct"] == 20
    assert result["warning_level"] == "RED"
    assert result["market_warning"]["data_status"] == "shadow"
    assert result["effective_gate_source"] == "legacy_market_risk"


def _register_active_warning_models(repository, version="model-v2"):
    for registered_market in Market:
        for horizon in ("1d", "3d"):
            repository.register_model(
                {
                    "model_version": version,
                    "market": registered_market,
                    "horizon": horizon,
                    "feature_version": "market-warning-v2",
                    "calibration_version": "platt-v2",
                    "training_cutoff": "2026-07-31",
                    "artifact_path": f"/tmp/{version}/{registered_market.value}/{horizon}.joblib",
                    "artifact_sha256": "0" * 64,
                    "metrics": {},
                    "base_rate": 0.01,
                    "active": True,
                }
            )


def _persist_warning(
    db_path,
    *,
    market,
    at,
    status="fresh",
    level="ORANGE",
    activate_models=True,
):
    repository = SQLiteWarningRepository(db_path)
    if activate_models:
        _register_active_warning_models(repository)
    snapshot = FeatureSnapshot(
        market=market,
        as_of_time=_dt.datetime.fromisoformat(at),
        session_slot="intraday-0935",
        feature_version="market-warning-v2",
        features={"market_phase": "FIRST_SHOCK"},
        evidence=(),
        data_quality=status,
        reliability_grade="A" if status == "fresh" else "C",
        source_times={"fixture": _dt.datetime.fromisoformat(at)},
    )
    quant = QuantRiskAssessment(
        crash_1d_probability=0.05,
        crash_3d_probability=0.10,
        market_phase=MarketPhase.FIRST_SHOCK,
        base_rate_1d=0.01,
        base_rate_3d=0.02,
        reliability_grade="A",
        model_version="model-v2",
        calibration_version="platt-v2",
        top_contributors=(),
    )
    actions = {"ORANGE": ("CONDITIONAL", 3.0), "RED": ("WAIT", 0.0)}
    gate, cap = actions[level]
    decision = FinalWarningDecision(
        baseline_level=level,
        final_level=level,
        state_transition=f"INITIAL_{level}",
        entry_gate=gate,
        new_position_cap_pct=cap,
        holding_action="HOLD_OR_REDUCE" if level == "ORANGE" else "REDUCE",
        push_required=True,
        decision_reasons=("test warning",),
        data_status=status,
    )
    snapshot_id = repository.save_feature_snapshot(snapshot)
    prediction_ids = repository.save_predictions(snapshot_id, quant)
    repository.save_decision(snapshot_id, prediction_ids, None, decision)


def _persist_rule_warning(
    db_path,
    *,
    at="2026-08-03T01:35:00+00:00",
    level="ORANGE",
    reliability="A",
    status="fresh",
    activation="notify",
    assessment_digest="a" * 64,
    registry_digest="a" * 64,
):
    repository = SQLiteWarningRepository(db_path)
    repository.register_rule_engine(
        {
            "engine_version": "rule-v1.0.0",
            "market": Market.A_SHARE,
            "manifest_sha256": registry_digest,
            "metrics": {},
        }
    )
    repository.activate_rule_engine("rule-v1.0.0", "notify")
    if activation == "gate":
        repository.activate_rule_engine("rule-v1.0.0", "gate")
    timestamp = _dt.datetime.fromisoformat(at)
    snapshot = FeatureSnapshot(
        market=Market.A_SHARE,
        as_of_time=timestamp,
        session_slot="intraday-0935",
        feature_version="market-warning-v2",
        features={"market_phase": "FIRST_SHOCK"},
        evidence=(),
        data_quality=status,
        reliability_grade=reliability,
        source_times={"fixture": timestamp},
    )
    actions = {
        "GREEN": ("OPEN", 100.0),
        "YELLOW": ("OPEN", 100.0),
        "ORANGE": ("CONDITIONAL", 3.0),
        "RED": ("WAIT", 0.0),
    }
    gate, cap = actions[level]
    assessment = RuleRiskAssessment(
        market=Market.A_SHARE,
        as_of_time=timestamp,
        engine_version="rule-v1.0.0",
        manifest_sha256=assessment_digest,
        risk_level=level,
        risk_score=5.0 if level in {"ORANGE", "RED"} else 1.0,
        market_phase=MarketPhase.FIRST_SHOCK,
        triggered_rules=(),
        missing_optional_groups=(),
        reliability_grade=reliability,
        evaluation_latency_ms=5.0,
    )
    decision = FinalWarningDecision(
        baseline_level=level,
        final_level=level,
        state_transition=f"INITIAL_{level}",
        entry_gate=gate,
        new_position_cap_pct=cap,
        holding_action="REDUCE" if level == "RED" else "HOLD_OR_REDUCE",
        push_required=level in {"ORANGE", "RED"},
        decision_reasons=("rule warning",),
        data_status=status,
        decision_source=DecisionSource.RULE_V1,
    )
    snapshot_id = repository.save_feature_snapshot(snapshot)
    assessment_id = repository.save_rule_assessment(snapshot_id, assessment)
    repository.save_decision(
        snapshot_id,
        (),
        None,
        decision,
        rule_assessment_id=assessment_id,
    )


def test_warning_loader_ignores_decisions_from_inactive_model_sets():
    with _tmp_dir() as tmp_path:
        db_path = tmp_path / "risk.db"
        _persist_warning(
            db_path,
            market=Market.A_SHARE,
            at="2026-08-03T01:35:00+00:00",
            activate_models=False,
        )

        warning = load_market_warning_for_ticker(
            "300308",
            "2026-08-03",
            db_path,
            analysis_time="2026-08-03T09:40:00+08:00",
        )

        assert warning is None


def test_rule_notify_activation_is_visible_but_never_changes_hard_gate():
    with _tmp_dir() as tmp_path:
        db_path = tmp_path / "risk.db"
        _persist_rule_warning(db_path, activation="notify", level="RED")

        warning = load_market_warning_for_ticker(
            "300308",
            "2026-08-03",
            db_path,
            analysis_time="2026-08-03T09:40:00+08:00",
        )
        effective = compose_effective_market_gate(
            {"market": "a_share", "entry_gate": "OPEN", "position_cap_pct": 20},
            warning,
            "a_share",
        )

        assert warning["decision_source"] == "rule_v1"
        assert warning["engine_version"] == "rule-v1.0.0"
        assert warning["notification_only"] is True
        assert warning["gate_applicable"] is False
        assert warning.get("probabilities") is None
        assert effective["entry_gate"] == "OPEN"
        assert effective["position_cap_pct"] == 20
        assert effective["warning_level"] == "RED"


def test_rule_gate_activation_constrains_only_fresh_reliable_orange_or_red():
    for level, expected_gate, expected_cap in (
        ("ORANGE", "CONDITIONAL", 3.0),
        ("RED", "WAIT", 0.0),
    ):
        with _tmp_dir() as tmp_path:
            db_path = tmp_path / "risk.db"
            _persist_rule_warning(db_path, activation="gate", level=level)

            warning = load_market_warning_for_ticker(
                "300308",
                "2026-08-03",
                db_path,
                analysis_time="2026-08-03T09:40:00+08:00",
            )
            effective = compose_effective_market_gate(
                {"market": "a_share", "entry_gate": "OPEN", "position_cap_pct": 20},
                warning,
                "a_share",
            )

            assert warning["notification_only"] is False
            assert warning["gate_applicable"] is True
            assert effective["entry_gate"] == expected_gate
            assert effective["position_cap_pct"] == expected_cap


def test_rule_gate_ignores_green_low_reliability_stale_and_checksum_mismatch():
    cases = (
        {"level": "GREEN"},
        {"level": "ORANGE", "reliability": "C"},
        {"level": "ORANGE", "status": "stale"},
        {"level": "ORANGE", "assessment_digest": "b" * 64},
    )
    for changes in cases:
        with _tmp_dir() as tmp_path:
            db_path = tmp_path / "risk.db"
            _persist_rule_warning(db_path, activation="gate", **changes)

            warning = load_market_warning_for_ticker(
                "300308",
                "2026-08-03",
                db_path,
                analysis_time="2026-08-03T09:40:00+08:00",
            )

            if changes.get("assessment_digest"):
                assert warning is None
            else:
                assert warning["gate_applicable"] is False
                assert warning["notification_only"] is True


def test_stale_a_share_warning_fails_closed_but_shadow_us_does_not():
    with _tmp_dir() as tmp_path:
        db_path = tmp_path / "risk.db"
        _persist_warning(
            db_path,
            market=Market.A_SHARE,
            at="2026-08-03T01:35:00+00:00",
        )
        stale = load_market_warning_for_ticker(
            "300308",
            "2026-08-03",
            db_path,
            analysis_time="2026-08-03T10:00:00+08:00",
        )
        assert stale["warning_level"] == "UNKNOWN"
        assert stale["entry_gate"] == "WAIT"
        assert stale["position_cap_pct"] == 0
        assert "陈旧" in "；".join(stale["reasons"])
        assert "市场压力" not in "；".join(stale["reasons"])

    with _tmp_dir() as tmp_path:
        db_path = tmp_path / "risk.db"
        _persist_warning(
            db_path,
            market=Market.US,
            at="2026-08-03T13:35:00+00:00",
            status="shadow",
            level="RED",
        )
        shadow = load_market_warning_for_ticker(
            "NVDA",
            "2026-08-03",
            db_path,
            analysis_time="2026-08-03T11:00:00-04:00",
        )
        assert shadow["warning_level"] == "RED"
        assert shadow["data_status"] == "shadow"


def test_warning_loader_never_reads_a_later_same_day_decision():
    with _tmp_dir() as tmp_path:
        db_path = tmp_path / "risk.db"
        _persist_warning(
            db_path,
            market=Market.A_SHARE,
            at="2026-08-03T01:35:00+00:00",
            level="ORANGE",
        )
        _persist_warning(
            db_path,
            market=Market.A_SHARE,
            at="2026-08-03T06:55:00+00:00",
            level="RED",
        )

        warning = load_market_warning_for_ticker(
            "300308",
            "2026-08-03",
            db_path,
            analysis_time="2026-08-03T09:40:00+08:00",
        )

        assert warning["warning_level"] == "ORANGE"
        assert warning["as_of_time"] == "2026-08-03T01:35:00+00:00"

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001 — 自跑器需把异常也显形
            failed += 1
            print(f"  ✗ {fn.__name__}: [{type(e).__name__}] {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
