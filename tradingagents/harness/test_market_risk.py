"""市场风险快照的确定性行为回归测试。

运行：python tradingagents/harness/test_market_risk.py
（项目自跑风格，不依赖 pytest；原 pytest fixture tmp_path/monkeypatch/capsys 已就地手写。）
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

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
