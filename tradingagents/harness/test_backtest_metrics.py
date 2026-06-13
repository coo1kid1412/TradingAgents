"""锁回测度量口径：signed PnL 记账 + horizon 缩放命中带。

修两个 2026-06-13 周报暴露的测量 bug：
1. 收益未按方向取符号 → 成功看空(避开-10%下跌)被记成-10%亏损 → HOLD/SELL 全线"赔钱"假象
2. 命中带固定±2% → 高波动 regime 下 5 日窗口 HOLD 几乎全踩空

运行：python tradingagents/harness/test_backtest_metrics.py
"""
from tradingagents.harness.truth_fetcher import _signed_pnl, _direction_hit, _hit_band
from tradingagents.harness.backtest import _compute_group_metric


def test_signed_pnl_signs_by_direction():
    assert _signed_pnl("long", 8.0) == 8.0           # 做多涨8 → +8
    assert _signed_pnl("long", -5.0) == -5.0         # 做多跌5 → -5
    assert _signed_pnl("short", -10.0) == 10.0       # 看空判对(跌10) → +10（核心修复）
    assert _signed_pnl("short", 6.0) == -6.0         # 看空判错(涨6) → -6
    assert _signed_pnl("neutral", 5.0) == 5.0        # HOLD = 持有 → 拿涨跌幅
    assert _signed_pnl(None, 5.0) is None
    assert _signed_pnl("short", None) is None
    print("✓ signed PnL 按方向取符号（看空判对记正收益）")


def test_hit_band_scales_with_horizon():
    assert _hit_band("T") == 2.0 and _hit_band("T+5") == 5.0 and _hit_band("T+30") == 10.0
    # 涨4%：T 带±2 踩空、T+5 带±5 命中
    assert _direction_hit("neutral", 4.0, "T") == 0
    assert _direction_hit("neutral", 4.0, "T+5") == 1
    # 做多涨3%：T 达标(>2)、T+30 不达标(>10 才算)
    assert _direction_hit("long", 3.0, "T") == 1
    assert _direction_hit("long", 3.0, "T+30") == 0
    # 看空跌12%：T+30 命中
    assert _direction_hit("short", -12.0, "T+30") == 1
    print("✓ 命中带按 horizon 缩放")


def test_group_metrics_uses_signed_pnl():
    """correct short 不再拖累期望；期望 = signed PnL 均值。"""
    rows = [
        # 看空判对：股票跌10%，signed_pnl=+10，direction_hit=1
        {"direction_hit": 1, "signed_pnl_pct": 10.0, "realized_return_pct": -10.0},
        # 做多判对：涨8，signed=+8
        {"direction_hit": 1, "signed_pnl_pct": 8.0, "realized_return_pct": 8.0},
        # 做多判错：跌3，signed=-3
        {"direction_hit": 0, "signed_pnl_pct": -3.0, "realized_return_pct": -3.0},
    ]
    m = _compute_group_metric(rows)
    assert m["direction_hit_rate"] == round(2 / 3, 4)
    assert m["avg_return_correct"] == 9.0          # (10+8)/2，不再被 -10 拉低
    assert m["avg_return_wrong"] == -3.0
    assert round(m["expectation"], 4) == round((10 + 8 - 3) / 3, 4)   # signed PnL 直接均值
    print("✓ 切片期望用 signed PnL（看空判对贡献正收益）")


def test_group_metrics_fallback_when_signed_missing():
    """旧行无 signed_pnl 时回退 realized_return，不 None 污染。"""
    rows = [
        {"direction_hit": 1, "signed_pnl_pct": None, "realized_return_pct": 5.0},
        {"direction_hit": 0, "signed_pnl_pct": None, "realized_return_pct": -2.0},
    ]
    m = _compute_group_metric(rows)
    assert m["expectation"] == round((5 - 2) / 2, 4)
    print("✓ signed 缺失回退 realized_return")


if __name__ == "__main__":
    test_signed_pnl_signs_by_direction()
    test_hit_band_scales_with_horizon()
    test_group_metrics_uses_signed_pnl()
    test_group_metrics_fallback_when_signed_missing()
    print("\n全部 4 组通过 ✅")
