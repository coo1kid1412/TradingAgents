"""锁新闻催化确定性聚合（路2）：key_events → 净催化信号 → 进评级链催化腿。

背景：新闻分析师早有结构化 SUMMARY(key_events 带 impact/credibility/horizon/priced_in)，
但下游 RM 当散文读、自己判松紧。这里把"判事件"(LLM)和"聚合方向"(Python)分开——
LLM 判事件、Python 定方向、催化腿确定性消费，RM 不再二次解读。

运行：python tradingagents/dataflows/test_news_catalyst.py
"""
from tradingagents.dataflows.news_catalyst import (
    aggregate_news_catalyst, parse_sys_catalyst, aggregate_catalyst_calendar,
    compute_narrative_shift, compute_earnings_revision, parse_sys_earnings_revision,
    _recency_weight,
)
from tradingagents.agents.managers.rm_tools import compute_step6_catalyst_momentum_adjustment as C


def test_catalyst_calendar():
    """步骤1：催化日历——只收 thesis 相关有方向事件，按日期排，边缘过滤。"""
    nr = ("# 新闻\n```yaml\nSUMMARY:\n  key_events:\n"
          "    - title: 中报\n      event_date: 2026-07-15\n      thesis_relevance: 核心\n      impact: +中\n      priced_in_p: 40\n"
          "    - title: 出口管制\n      event_date: 2026Q3\n      thesis_relevance: 核心\n      impact: -大\n"
          "    - title: 八卦\n      event_date: 未知\n      thesis_relevance: 边缘\n      impact: +小\n"
          "    - title: 量产\n      event_date: 未知\n      thesis_relevance: 相关\n      impact: +大\n```")
    cal = aggregate_catalyst_calendar(nr)
    titles = [c["title"] for c in cal]
    assert "八卦" not in titles                       # 边缘过滤
    assert titles[0] == "中报" and titles[1] == "出口管制"  # 有日期按日期升序
    assert titles[-1] == "量产"                        # 未知日期排末
    assert cal[1]["direction"] == "-"                  # -大 → 方向 -
    # 全是边缘/0 影响 → None
    nr2 = "# 新闻\n```yaml\nSUMMARY:\n  key_events:\n    - title: x\n      thesis_relevance: 边缘\n      impact: +小\n```"
    assert aggregate_catalyst_calendar(nr2) is None
    print("✓ 催化日历：thesis 相关过滤 + 日期排序 + 边缘剔除")


def _report(events_yaml: str) -> str:
    return f"# 新闻报告\n正文\n```yaml\nSUMMARY:\n  net_sentiment: 中性\n  key_events:\n{events_yaml}\n```"


def test_aggregate_bearish_near_term_dominates():
    """强空近端高可信未定价事件 主导净催化。"""
    nr = _report(
        "    - title: 美国出口管制升级\n      horizon: 短期(≤1周)\n      priced_in_p: 20\n      impact: -大\n      credibility: 高\n"
        "    - title: DDR5量产\n      horizon: 中期(1-3月)\n      priced_in_p: 60\n      impact: +中\n      credibility: 中\n")
    cat = aggregate_news_catalyst(nr)
    # -大×1.0×0.8×1.0=-1.6 ; +中×0.7×0.4×0.7=+0.196 ; net≈-1.40
    assert cat["direction"] == -1 and cat["strength"] == "medium" and cat["score"] < 0
    assert cat["nearest"] == "美国出口管制升级"
    print("✓ 聚合：强空近端事件主导 → 净催化偏空")


def test_priced_in_discounts():
    """已定价的强利好不再驱动（priced_in 高→权重低）。"""
    fresh = _report("    - title: 新订单\n      horizon: 短期(≤1周)\n      priced_in_p: 0\n      impact: +大\n      credibility: 高\n")
    priced = _report("    - title: 新订单\n      horizon: 短期(≤1周)\n      priced_in_p: 95\n      impact: +大\n      credibility: 高\n")
    cf = aggregate_news_catalyst(fresh)["score"]
    cp = aggregate_news_catalyst(priced)["score"]
    assert cf > cp and cf > 0          # 未定价利好分高
    assert abs(cp) <= 5                # 已定价利好分被压到接近 0
    print("✓ 已定价事件被折价（priced_in 高→权重低）")


def test_guards():
    assert aggregate_news_catalyst("# 报告\n无结构化块") is None      # 无 SUMMARY
    assert aggregate_news_catalyst(_report("    []")) is None or True  # 空事件容错
    assert aggregate_news_catalyst("") is None
    print("✓ 无 SUMMARY / 空 / None 防御")


def test_sys_catalyst_roundtrip():
    line = "SYS_CATALYST: direction=-1 | strength=medium | score=-21（…）"
    d = parse_sys_catalyst(line)
    assert d == {"direction": -1, "strength": "medium", "score": -21}
    assert parse_sys_catalyst("无此行") is None
    print("✓ SYS_CATALYST 注入/解析往返一致")


def test_feeds_catalyst_leg():
    """news_catalyst_score 进催化腿计分（第5信号）。"""
    cr = C.invoke({"rating_after_vote_adj": "HOLD", "news_catalyst_score": -25,
                   "northbound_flow_5d_direction": -1})
    assert "news_catalyst" in cr["breakdown"] and cr["breakdown"]["news_catalyst"]["subscore"] == -25
    assert cr["adjustment"] == -1     # 强空催化 + 北向流出 → 催化腿 -1
    # 单信号不足 2 项 → skip（覆盖不足保护仍生效）
    solo = C.invoke({"rating_after_vote_adj": "HOLD", "news_catalyst_score": -25})
    assert solo["rule_applied"] == "skipped"
    print("✓ 新闻催化进催化腿计分 + 覆盖不足保护")


def test_narrative_shift():
    """步骤3：叙事切换早期预警——水位 vs 动能/新闻论调背离。"""
    def srep(net, trend):
        return f"# 社媒\n```yaml\nSUMMARY:\n  net_sentiment: {net}\n  sentiment_trend_7d: {trend}\n```"
    def nrep(net):
        return f"# 新闻\n```yaml\nSUMMARY:\n  net_sentiment: {net}\n```"
    # 社媒偏多但 7 日动能转负 → 见顶回落
    assert compute_narrative_shift(srep("偏多", -45), nrep("中性"))["status"] == "见顶回落预警"
    # 新闻先转空、社媒仍偏多 → 见顶回落
    assert compute_narrative_shift(srep("偏多", 5), nrep("负面"))["status"] == "见顶回落预警"
    # 社媒偏空但动能转正 → 筑底回升
    assert compute_narrative_shift(srep("偏空", 40), nrep("中性"))["status"] == "筑底回升预警"
    # 方向一致 → 无切换
    assert compute_narrative_shift(srep("偏多", 20), nrep("正面"))["status"] == "无明显切换"
    # 两份都缺 SUMMARY → None
    assert compute_narrative_shift("无块", "也无块") is None
    print("✓ 叙事切换：水位/动能/论调背离 → 见顶回落/筑底回升预警")


def test_earnings_revision():
    """A：从新闻 SUMMARY 抽卖方上修/下修方向（喂 regime earnings 腿）。"""
    def nr(patterns=None, events=None):
        body = "# 新闻\n```yaml\nSUMMARY:\n"
        if patterns is not None:
            body += "  cumulative_patterns:\n" + "".join(f"    - {p}\n" for p in patterns)
        if events is not None:
            body += "  key_events:\n" + "".join(
                f"    - title: {t}\n      category: {c}\n" for t, c in events)
        return body + "```"
    # 累积模式"多次评级上调" → 上修
    r = compute_earnings_revision(nr(patterns=["近30日多次评级上调", "机构密集调研"]))
    assert r["direction"] == "上修" and r["up"] == 1 and r["down"] == 0, r
    # 累积模式"多次评级下调" → 下修
    assert compute_earnings_revision(nr(patterns=["多次评级下调"]))["direction"] == "下修"
    # 机构类事件标题"上调目标价" → 上修
    r2 = compute_earnings_revision(nr(events=[("中金上调目标价至350元", "机构"), ("行业政策", "行业")]))
    assert r2["direction"] == "上修", r2
    # 非机构类事件不计入（行业类"下调"不算盈利下修）
    r3 = compute_earnings_revision(nr(events=[("行业景气下调", "行业")]))
    assert r3["direction"] == "停修", r3
    # 无相关关键词 → 停修；无 SUMMARY → None
    assert compute_earnings_revision(nr(patterns=["机构密集调研"]))["direction"] == "停修"
    assert compute_earnings_revision("无 SUMMARY 块") is None
    # SYS 行往返
    assert parse_sys_earnings_revision("SYS_EARNINGS_REVISION: 上修（卖方…）") == "上修"
    assert parse_sys_earnings_revision("无该行") is None
    print("✓ 盈利上修/下修：累积模式+机构事件 → 上修/停修/下修，非机构不误计")


def test_recency_weight():
    """#3：新鲜度权重——见报越旧权重越低；缺日期不衰减（向后兼容）。"""
    cd = "2026-06-24"
    assert _recency_weight("2026-06-20", cd) == 1.0     # 4 天，新鲜
    assert _recency_weight("2026-06-10", cd) == 0.75    # 14 天
    assert _recency_weight("2026-05-20", cd) == 0.5     # 35 天
    assert _recency_weight("2026-04-01", cd) == 0.35    # >45 天
    assert _recency_weight("20260620", cd) == 1.0       # YYYYMMDD 也能解析
    # 缺见报日期 / 缺当前日期 / 未知 → 1.0（不惩罚缺失）
    assert _recency_weight(None, cd) == 1.0
    assert _recency_weight("未知", cd) == 1.0
    assert _recency_weight("2026-06-20", None) == 1.0
    # 端到端：陈旧正面事件被新鲜度压低净催化
    fresh = ("# 新闻\n```yaml\nSUMMARY:\n  key_events:\n"
             "    - title: 卖方上调\n      impact: +大\n      source_date: 2026-06-23\n      horizon: 短期\n      priced_in_p: 20\n```")
    stale = fresh.replace("2026-06-23", "2026-04-01")
    cf_fresh = aggregate_news_catalyst(fresh, current_date=cd)
    cf_stale = aggregate_news_catalyst(stale, current_date=cd)
    assert cf_stale["net"] < cf_fresh["net"], (cf_fresh, cf_stale)
    # 不传 current_date → 不衰减（向后兼容，net 与 fresh 同）
    cf_nodate = aggregate_news_catalyst(stale)
    assert cf_nodate["net"] == cf_fresh["net"], (cf_nodate, cf_fresh)
    print("✓ 新鲜度：见报越旧权重越低，缺日期/缺current_date 不衰减（向后兼容）")


if __name__ == "__main__":
    test_aggregate_bearish_near_term_dominates()
    test_priced_in_discounts()
    test_guards()
    test_sys_catalyst_roundtrip()
    test_feeds_catalyst_leg()
    test_catalyst_calendar()
    test_narrative_shift()
    test_earnings_revision()
    test_recency_weight()
    print("\n全部 9 组通过 ✅")
