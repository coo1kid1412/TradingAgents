"""锁 cyq_perf 筹码分布接入：winner_rate 解析 + 散户信号筹码口径优先。

背景：项目原散户指标 retail_buy_amount_rate 口径不可靠（实测 6-10% 远低应有值）、
股东户数季度滞后。2000 档解锁 cyq_perf 后用 winner_rate（获利盘%，日频可靠）替代。

运行：python tradingagents/dataflows/test_chip_distribution.py
"""
import pandas as pd
from unittest.mock import patch

from tradingagents.dataflows import tushare_vendor as tv
from tradingagents.dataflows.capital_flow_utils import (
    compute_retail_concentration_signal,
    assemble_capital_flow_metrics,
)


def _chip_df():
    return pd.DataFrame([
        {"trade_date": f"202606{d:02d}", "winner_rate": wr, "weight_avg": 486.0, "cost_50pct": 489.0}
        for d, wr in zip([3, 4, 5, 6, 9, 10, 11, 12], [55, 52, 48, 45, 42, 40, 41.59, 36.77])
    ])


def test_chip_parse():
    with patch.object(tv, "_fetch_cached", return_value=_chip_df()), patch.object(tv, "_get_tushare_api"):
        chip = tv.get_chip_distribution("603986", "2026-06-12")
    assert chip["winner_rate_pct"] == 36.77
    assert chip["winner_rate_chg_5d"] == round(36.77 - 48, 2)   # 较 5 根前(index -6=48)
    assert chip["weight_avg_cost"] == 486.0
    # 行数不足 6 → chg 为 None
    with patch.object(tv, "_fetch_cached", return_value=_chip_df().tail(3)), patch.object(tv, "_get_tushare_api"):
        chip2 = tv.get_chip_distribution("603986", "2026-06-12")
    assert chip2["winner_rate_chg_5d"] is None
    # 接口缺列 → None
    with patch.object(tv, "_fetch_cached", return_value=pd.DataFrame([{"trade_date": "20260612"}])), patch.object(tv, "_get_tushare_api"):
        assert tv.get_chip_distribution("603986", "2026-06-12") is None
    print("✓ get_chip_distribution 解析 + 5日变化 + 缺数据防御")


def test_retail_signal_chip_first():
    # 筹码口径优先：主力派发(streak≤-3) + 获利盘低(套牢) → 散户高接盘
    assert compute_retail_concentration_signal(None, -7, winner_rate_pct=36.77) == "散户高接盘"
    # 获利盘高 → 没套牢 → 中性（即便主力派发）
    assert compute_retail_concentration_signal(None, -7, winner_rate_pct=85.0) == "中性"
    # 主力没派发(streak=0) → 中性（套牢但非派发场景）
    assert compute_retail_concentration_signal(None, 0, winner_rate_pct=36.77) == "中性"
    # winner_rate 优先于毛买占比（坏口径）
    assert compute_retail_concentration_signal(70.0, -7, winner_rate_pct=36.77) == "散户高接盘"
    # 无 winner_rate → 退回毛买占比口径
    assert compute_retail_concentration_signal(70.0, -7) == "散户高接盘"
    # 全缺 → None
    assert compute_retail_concentration_signal(None, None) is None
    print("✓ 散户信号筹码口径优先 + 兜底退化")


def test_assemble_wires_chip():
    m = assemble_capital_flow_metrics(
        chip_metrics={"winner_rate_pct": 36.77, "winner_rate_chg_5d": -8.2, "weight_avg_cost": 486.0})
    assert m["winner_rate_pct"] == 36.77
    assert m["winner_rate_chg_5d"] == -8.2
    assert m["chip_weight_avg_cost"] == 486.0
    # 无 chip → 字段为 None，不崩
    m2 = assemble_capital_flow_metrics()
    assert m2["winner_rate_pct"] is None
    print("✓ assemble 接入筹码字段 + 缺失防御")


if __name__ == "__main__":
    test_chip_parse()
    test_retail_signal_chip_first()
    test_assemble_wires_chip()
    print("\n全部 3 组通过 ✅")
