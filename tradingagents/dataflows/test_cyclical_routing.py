"""锁周期股路由：识别清单 / SYS_CYCLICAL 机读行 / normalized EPS / regime 周期反转。

背景（林奇铁律）：周期股顶部 TTM PE 最低（该卖）、谷底最高（该买），TTM 口径整个反着。
- 口径：normalized EPS = 年度 ROE 十年中位 ÷100 × 最新 BPS（卖方 mid-cycle 标准做法）
- 周期位置：当前年化 ROE 在 ≤10 个年度 ROE 的分位（≥0.8 top / ≤0.2 trough）
- regime 反转：strong 顶部高增速不投 +1（防"顶部判 ride 下不了车"）、
  谷底负增不投 -1（防"谷底判 discipline 封死 BUY"）

运行：python tradingagents/dataflows/test_cyclical_routing.py
"""
import pandas as pd

from tradingagents.dataflows.profile_calc import (
    detect_cyclical,
    parse_sys_cyclical,
    compute_valuation_regime,
    cyclical_target_weights,
)


def test_cyclical_target_weights_slide_by_position():
    """正常化 vs 成长前瞻 目标价权重按周期位置滑动（治兆易 230↔723 摆动）。"""
    assert cyclical_target_weights("top") == (0.7, 0.3)     # 顶部:正常化主导(谨慎)
    assert cyclical_target_weights("mid") == (0.5, 0.5)
    assert cyclical_target_weights("trough") == (0.3, 0.7)  # 谷底:成长主导(乐观)
    assert cyclical_target_weights("数据不足") == (0.6, 0.4)  # 缺位置:偏谨慎
    assert cyclical_target_weights(None) == (0.6, 0.4)
    # 权重和恒为 1（混合不漏权）
    for pos in ("top", "mid", "trough", "数据不足", None):
        wn, wg = cyclical_target_weights(pos)
        assert abs(wn + wg - 1.0) < 1e-9
    print("✓ 周期目标价权重按位置滑动，和恒=1")
from tradingagents.dataflows.tushare_vendor import _format_cyclical_line


def _fina(latest_roe_cum: float, latest_end: str = "20260331") -> pd.DataFrame:
    roes = [4.0, 8.0, 15.0, 22.0, 10.0, 3.0, 6.0, 12.0, 20.0, 9.0]  # 两轮周期，中位 9.5
    rows = [{"end_date": f"{2016+i}1231", "roe": r, "bps": 10.0 + i * 0.5, "eps": r / 10}
            for i, r in enumerate(roes)]
    rows.append({"end_date": latest_end, "roe": latest_roe_cum, "bps": 15.0, "eps": 0.6})
    return pd.DataFrame(rows)


def test_detect_cyclical():
    assert detect_cyclical("普钢", "宝钢股份") == "strong"        # 行业关键词
    assert detect_cyclical("元器件", "京东方A") == "strong"       # 科技周期名单（行业太粗靠名单）
    assert detect_cyclical("半导体", "兆易创新") == "strong"      # 存储
    assert detect_cyclical("半导体", "长电科技") == "semi"        # 封测半周期
    assert detect_cyclical("半导体", "澜起科技") is None          # secular 成长不误伤
    assert detect_cyclical(None, None) is None
    print("✓ 周期识别：行业关键词 + 科技名单，secular 不误伤")


def test_cyclical_line_top_and_trough():
    # Q1 累计 6% → 年化 24% → 高于全部年度值 → top
    line = _format_cyclical_line("元器件", "京东方A", _fina(6.0), close_price=4.5)
    d = parse_sys_cyclical(line)
    assert d["class"] == "strong" and d["position"] == "top", line
    assert abs(d["roe_10y_median"] - 9.5) < 0.1
    # normalized EPS = 9.5% × 15.0 = 1.43（一分钱不差）
    assert abs(d["normalized_eps"] - 1.43) < 0.01, d
    assert d["pe_on_normalized"] is not None
    # Q1 累计 0.5% → 年化 2% → trough
    d2 = parse_sys_cyclical(_format_cyclical_line("元器件", "京东方A", _fina(0.5), 4.5))
    assert d2["position"] == "trough"
    print("✓ 顶/谷识别 + normalized EPS（ROE十年中位×BPS）")


def test_cyclical_line_guards():
    assert _format_cyclical_line("半导体", "澜起科技", _fina(6.0), 4.5) == ""   # 非周期不发射
    short = _fina(6.0).tail(3)
    l3 = _format_cyclical_line("普钢", "宝钢股份", short, 4.5)
    assert "数据不足" in l3 and parse_sys_cyclical(l3)["class"] == "strong"     # 新上市降级
    assert _format_cyclical_line("普钢", "宝钢股份", None, 4.5) == "【SYS_CYCLICAL｜tushare】 class=strong | position=数据不足"
    print("✓ 非周期不发射 / 样本不足降级 / 空数据防御")


def test_regime_cyclical_inversion():
    base = dict(momentum_score=80, capital_flow_score=70, net_profit_growth=1.2,
                theme_stage_inferred="acceleration")
    # 非周期：高增速 earnings +1
    assert compute_valuation_regime(**base)["legs"]["earnings"] == 1
    # 强周期顶部：高增速是周期顶部现象 → 压 0
    top = compute_valuation_regime(**base, cyclical_class="strong", roe_pct_rank_10y=0.92)
    assert top["legs"]["earnings"] == 0 and "顶部" in top["reasoning"]
    # 强周期谷底：负增长是周期常态 → 抬 0
    trough = compute_valuation_regime(momentum_score=30, capital_flow_score=50,
                                      net_profit_growth=-0.3, theme_stage_inferred="none",
                                      cyclical_class="strong", roe_pct_rank_10y=0.10)
    assert trough["legs"]["earnings"] == 0 and "谷底" in trough["reasoning"]
    # 半周期保留成长β语义，不反转
    semi = compute_valuation_regime(**base, cyclical_class="semi", roe_pct_rank_10y=0.92)
    assert semi["legs"]["earnings"] == 1
    # 周期中段不动
    mid = compute_valuation_regime(**base, cyclical_class="strong", roe_pct_rank_10y=0.5)
    assert mid["legs"]["earnings"] == 1
    print("✓ regime 周期反转：顶压0/谷抬0/半周期与中段不动")


if __name__ == "__main__":
    test_detect_cyclical()
    test_cyclical_line_top_and_trough()
    test_cyclical_line_guards()
    test_regime_cyclical_inversion()
    test_cyclical_target_weights_slide_by_position()
    print("\n全部 5 组通过 ✅")
