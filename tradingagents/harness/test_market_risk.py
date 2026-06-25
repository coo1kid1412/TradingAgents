"""市场风险快照的确定性行为回归测试。"""

import json

from tradingagents.harness.market_risk import (
    compute_market_risk_snapshot,
    infer_market,
    save_market_risk_snapshot,
    load_latest_market_risk_snapshot,
    load_market_risk_for_ticker,
)
from tradingagents.harness import market_risk_daily
from tradingagents.harness.market_risk_daily import is_market_trading_day, run_market_risk_daily, _send_feishu_message


def _prices(values):
    return [{"close": value} for value in values]


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


def test_snapshot_storage_is_unique_per_market_and_date(tmp_path):
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


def test_daily_runner_skips_closed_market_and_does_not_push(tmp_path):
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


def test_daily_runner_persists_and_push_failure_does_not_discard_snapshot(tmp_path):
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


def test_daily_runner_can_target_one_market_for_its_local_preopen(tmp_path):
    result = run_market_risk_daily(
        as_of_date="2026-06-25", db_path=tmp_path / "risk.db", markets=("a_share",),
        fetch_prices=lambda _ticker, _start, _end: _prices([100 + i for i in range(60)]),
        send_message=lambda _text: None,
    )

    assert set(result) == {"a_share"}


def test_feishu_sender_uses_open_api_credentials_when_webhook_is_absent(monkeypatch):
    """已有飞书应用凭证时不需要额外配置自定义机器人 webhook。"""
    monkeypatch.delenv("FEISHU_MARKET_RISK_WEBHOOK", raising=False)
    monkeypatch.setenv("FEISHU_APP_ID", "app-id")
    monkeypatch.setenv("FEISHU_APP_SECRET", "app-secret")
    monkeypatch.setenv("FEISHU_USER_OPEN_ID", "user-open-id")
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

    monkeypatch.setattr("tradingagents.harness.market_risk_daily._ur.urlopen", fake_urlopen)

    _send_feishu_message("市场风险测试")

    assert len(calls) == 2
    assert calls[0][1] == {"app_id": "app-id", "app_secret": "app-secret"}
    assert calls[1][1]["receive_id"] == "user-open-id"
    assert calls[1][1]["msg_type"] == "text"


def test_cli_treats_dry_run_as_success(monkeypatch, capsys):
    monkeypatch.setattr(
        market_risk_daily,
        "run_market_risk_daily",
        lambda **_kwargs: {"a_share": {"status": "dry_run", "push_status": "skipped"}},
    )
    monkeypatch.setattr("sys.argv", ["market_risk_daily", "--market", "a_share", "--dry-run"])

    assert market_risk_daily.main() == 0
    assert "a_share: dry_run / push=skipped" in capsys.readouterr().out
