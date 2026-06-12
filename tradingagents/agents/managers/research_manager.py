import logging
import json

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from tradingagents.agents.utils.agent_utils import build_instrument_context, RISK_DEBATE_PHRASING_RULES
from tradingagents.agents.managers.rm_tools import RM_TOOLS, RM_TOOLS_BY_NAME

logger = logging.getLogger(__name__)

# 工具调用循环上限——避免 LLM 反复调同一工具陷入死循环
# Step 6 评级终段已合并为 compute_step6_final_rating 一次调用（阈值/映射/拥挤/
# 升降档/叠加/极端防御一次合议），典型轮数 ~6-9，15 是宽裕缓冲
_MAX_TOOL_ITERATIONS = 15


def _run_tool_calling_loop(llm_with_tools, initial_messages, *,
                           tools_by_name=None, role="RM",
                           completion_token="RM_SUMMARY",
                           max_iterations=None, max_continuations=2):
    """执行 LLM 工具调用循环。返回 AIMessage，content 是所有迭代的 LLM 文本累积。

    退出条件不只是『不再调工具』，还要求**正文完整**：必须包含 completion_token
    （RM_SUMMARY / PM_SUMMARY，两者本就是 prompt 强制的收尾 YAML，天然的完成标记）。

    为什么：MiniMax 推理模型会把部分轮次的正文整段写进 <think> 块，被
    _strip_think_tags 剥成空串。两类真实事故：
    - 全空（603629 PM）：decision.md 缺失；
    - 截断（300394 RM）：首轮有正文、后续轮全空，thesis 停在 Step 4 中途，
      下游 PM 拿着没有评级的半截报告自己编了个 OVERWEIGHT 推送出去。
    截断比全空更隐蔽——只查空兜不住，必须查收尾标记。

    恢复策略：正文缺失/截断时带明确指令续写（工具照调，Step 6 的强制工具调用
    可能还没发生），最多 max_continuations 次；预算用尽仍全空则抛错显式失败，
    有部分正文则降级返回并 WARNING 留痕。
    """
    tools_by_name = tools_by_name if tools_by_name is not None else RM_TOOLS_BY_NAME
    max_iterations = max_iterations or _MAX_TOOL_ITERATIONS
    hard_cap = max_iterations + 2 * max_continuations + 2  # 续写也要算轮数，防死循环

    messages = list(initial_messages)
    cot_segments: list[str] = []
    continuations = 0
    iteration = 0

    while iteration < hard_cap:
        iteration += 1
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # 累积本轮 LLM 输出的文本（即使含 tool_calls，也常含 Step COT 片段）
        content = (response.content or "").strip()
        if content:
            cot_segments.append(content)

        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            logger.info("%s 第 %d 轮工具调用：%d 个工具", role, iteration, len(tool_calls))
            for tc in tool_calls:
                tool_name = tc.get("name")
                tool_args = tc.get("args", {})
                tool_id = tc.get("id", "")

                tool = tools_by_name.get(tool_name)
                if tool is None:
                    error_msg = f"未知工具：{tool_name}"
                    logger.warning(error_msg)
                    messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
                    continue

                try:
                    result = tool.invoke(tool_args)
                    result_str = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
                    logger.debug("%s 工具 %s 结果: %s", role, tool_name, result_str[:200])
                    messages.append(ToolMessage(content=result_str, tool_call_id=tool_id))
                except Exception as e:
                    error_msg = f"工具 {tool_name} 执行失败: {e}"
                    logger.warning(error_msg)
                    messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))

            # 工具轮数到达上限：要求收笔（仍允许它在续写检查中被抓回来）
            if iteration == max_iterations:
                logger.warning("%s 达到工具调用上限 (%d 轮)，要求直接收笔", role, max_iterations)
                messages.append(HumanMessage(
                    content="你已经调用足够多次工具了。请基于已有的工具结果直接写出最终报告，"
                            "**不要再调用任何工具**。"
                ))
            continue

        # —— 不再调工具：检查正文是否完整 ——
        joined = "\n\n".join(cot_segments)
        if joined.strip() and completion_token in joined:
            logger.info(
                "%s tool calling 循环结束（第 %d 轮，累积 %d 段，总长 %d 字符）",
                role, iteration, len(cot_segments), len(joined),
            )
            return AIMessage(content=joined)

        if continuations >= max_continuations:
            break

        continuations += 1
        reason = "为空" if not joined.strip() else f"截断（缺 {completion_token} 收尾）"
        logger.warning("%s 输出正文%s（疑似 think-only 被剥离），第 %d 次续写",
                       role, reason, continuations)
        messages.append(HumanMessage(
            content=f"你的报告正文{reason}——可能你把内容写进了思考过程。请**从中断处继续输出正文**，"
                    f"不要重复已输出的部分，不要输出思考标签；尚未完成的步骤继续完成"
                    f"（该调用的工具照常调用），最后必须以完整的 {completion_token} YAML 段收尾。"
        ))

    # —— 续写预算用尽 ——
    joined = "\n\n".join(cot_segments)
    if not joined.strip():
        raise RuntimeError(
            f"{role} 连续 {max_continuations + 1} 次输出空正文（think-only 剥离后无内容），"
            "中止本次分析——空 thesis 流向下游只会产出无依据的报告"
        )
    logger.warning("%s 续写预算用尽仍缺 %s 收尾，按现有 %d 字符正文降级返回（下游解析会标 warning）",
                   role, completion_token, len(joined))
    return AIMessage(content=joined)


def create_research_manager(llm):
    """改造 A：移除 memory 参数（invest_judge_memory 在当前工作流下永远为空，纯属装饰）。"""
    def research_manager_node(state) -> dict:
        rm_ticker = state["company_of_interest"]
        rm_trade_date = state.get("trade_date", "")
        instrument_context = build_instrument_context(rm_ticker, state.get("company_name", ""))
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        consensus_snapshot = state.get("consensus_snapshot", "")
        stock_profile = state.get("stock_profile", "")
        macro_context = state.get("macro_context", "")
        quant_score = state.get("quant_score", "")
        sector_comparison = state.get("sector_comparison", "")
        capital_flow_yaml = state.get("capital_flow_yaml", "")

        investment_debate_state = state["investment_debate_state"]

        prompt = f"""【语言要求】你必须使用中文撰写以下所有分析内容和回复。评级关键词（Buy/Overweight/Hold/Underweight/Sell）和股票代码可保留英文。

你是**投资研究总监（Head of Research）**，对标真实头部投研团队的 RM 角色。

## 你的角色定位（与裁判员的根本区别）

**真实 RM 不是"多空辩论的裁判员"，而是"独立研究员"**：
- 裁判员看双方哪边说得更有理 → 评级是公式产出
- 研究员自己看数据、建估值模型、判断业绩拐点 → 评级是综合判断产出

你的工作是：**走完整 8 步 COT 思考链路，最后给出主观但 COT 透明的评级**。多空辩论是"已识别的争议点清单"，给 PM 做仓位参考，**不主导评级方向**。

## 你的角色边界

- 只输出 thesis：评级 + 目标价区间 + 业绩拐点判断 + 风险清单
- 不输出执行细节：建仓价/止损价/仓位比例由 PM 决定
- 评级是综合判断，必须 COT 透明（能说清楚"为什么是这个评级而不是其他"）

## ⚠️ 数值计算必须调用工具（全局约束，下面每步不再重复说明）

你已绑定 11 个计算工具（见 `compute_*` 系列）。**凡涉及目标价 / 概率加权 / 偏离度 / 评级映射 / 三情景一致性 / 趋势叠加 / Bull-Bear 评分等数值，必须调用对应工具**，工具返回值直接采用，禁止心算或"调整修正"。引用数值时显式说明来源（"工具计算结果：__"）。

⚡ **批量调用（省往返，强制）**：**相互独立、不依赖彼此输出的工具调用，必须在同一轮一次性发起**（一条消息里放多个 tool_calls），不要一个一个排队调。典型可批量的场景：
- Step 4 的多种估值方法（pe_eps / peg 等彼此独立）→ 同一轮一起发；最后再单独调 weighted/overlap 汇总（它依赖前面的结果）。
- 辅助分析的 `compute_bull_bear_score` 与 `compute_conviction_calibration` 等无依赖项 → 同一轮发。
- 仅当 B 工具的输入确实来自 A 工具的返回值时（如汇总目标价依赖各方法目标价、rating_mapping 依赖目标价中位），才分轮串行。

各步骤对应工具速查：
- Step 4：`compute_pe_eps_target_price` / `compute_peg_target_price` / `compute_overlap_target_price` / `compute_weighted_target_price`
- Step 5：`compute_scenario_consistency_check`（强制）/ `compute_scenario_weighted_e`
- Step 6：`compute_step6_final_rating`（强制，一次合议：阈值/映射/regime 闸门/拥挤/升降档/趋势叠加/极端防御/不变量全在工具内）
- 辅助分析：`compute_bull_bear_score` / `compute_score_difference` / `compute_conviction_calibration` / `compute_odds_and_expected_return`

{instrument_context}

---

# 决策流程（必须严格按 8 步 COT 顺序完成）

## 第负一步：地雷排查清单（守门员一票否决）

⚠️ 命中任一条 → **直接 SELL，跳过所有后续步骤**：

| # | 地雷类型 | 触发信号 |
|---|---------|---------|
| L1 | ST/*ST 警示 | 股票名称含 "ST" / "*ST" |
| L2 | 退市风险 | 连续 2 年亏损 / 净资产为负 / 财报非标审计意见 |
| L3 | 财务造假嫌疑 | 审计师辞任 / 关联交易 > 营收 30% / 应收持续增速 > 营收 2 倍 |
| L4 | 大股东高质押+现金枯竭 | 控股质押 > 70% 且 货币资金/短期债务 < 0.5 |
| L5 | 监管立案 | 证监会/交易所立案调查 / 高管刑事立案 |
| L6 | 流动性枯竭 | 资产负债率 > 90% 且 流动比率 < 0.5 且 经营现金流 4 季为负 |
| L7 | 商誉爆雷 | 商誉/净资产 > 50% 且 行业景气下行 且 主营 -15% |
| L8 | 短期偿债危机 | 短期债务/(货币资金+短期投资) > 3 |
| L9 | 港股老千股 | 股价 < 1 HKD 或 历史合股 或 频繁配股缩股 |
| L10 | 借壳/重组失败 | 重大资产重组被否决 且 基本面严重恶化 |
| L11 | 资金链断裂前兆 | 银行抽贷 / 商票违约 / 评级机构连续下调 |

**输出**：每条 是/否 + 依据。**结论**：[未触发，继续 Step 1] / [触发 L__，直接 SELL]

---

## 第零步：吸收上下文（不打分，只读）

### 0.1 股票画像（由画像识别官提炼）

{stock_profile if stock_profile else "（画像缺失，按通用框架处理）"}

### 0.3 宏观上下文（由宏观策略师提炼）

{macro_context if macro_context else "（宏观上下文缺失，按中性环境处理）"}

**重点提取**：MACRO_CONTEXT.{{rate_cycle, liquidity, industry_macro_direction, premium_adjustment_pct}}

**用法**：
- Step 1 行业景气度判断必须叠加宏观顺/逆风修正（如行业本身上行但宏观强逆风 → 实际景气度降一档）
- Step 6 动态阈值计算时，stock_profile.THEMATIC_PREMIUM 已经吸收了宏观修正，你按其最终值使用即可

### 0.4 量化锚（由量化打分官 Python 确定性输出，无 LLM 主观空间）

{quant_score if quant_score else "（量化锚缺失，Step 6 趋势叠加路 / 极端背离防御跳过）"}

**重点提取**：
- QUANT_SCORE.composite（0-100 综合分）
- QUANT_SCORE.factor_scores.momentum（动量子分，0-100）
- 其他 5 类因子分项

**用法**：
- 这是**独立于 LLM 判断的量化锚**，由 Python 直接基于动量/价值/质量/成长/低波/反拥挤 6 因子打分得出
- **Step 6 趋势叠加路（在 `compute_step6_final_rating` 工具内部）**会用 composite + momentum 按 style 差异化调整评级（最多 ±1 档，不跨 HOLD）——你只需把这两个值如实填进工具入参
- **Step 6 极端背离防御（同一工具内部兜底）**——只在 composite ≤20 或 ≥80 这种极端值时触发强制调整
- factor_scores 中 <30 分的单项代表该维度有显著风险，Step 8 风险清单应当独立列出

**重点提取**：style / industry / VALUATION_METHOD.primary_method / target_pe_range / target_pb_range / data_completeness

**⚠️ stock_profile TRANSPARENCY 段（Layer 3 透明化标注）必读**：

stock_profile 末尾的 `TRANSPARENCY:` 段是 LLM 自填的"超共识程度标注"，按以下规则用于 **Step 6 评级 + Step 7 Conviction 校准**：

| TRANSPARENCY 字段 | 触发条件 | 你的应对 |
|------------------|---------|---------|
| `target_pe_high_vs_sell_side_pct` > +50 | 超卖方一致 +50% 以上 | 必须在 Step 7 Conviction 降一档（除非 stock_profile.premium_divergence_reason 列出 ≥2 条产业证据）|
| `target_pe_high_vs_sell_side_pct` > +100 | 超卖方一致 +100% | Conviction 强制最高"中"，且 Step 6 评级 COT 必须明确写"超共识溢价风险" |
| `theme_stage_llm_chosen` ≠ `theme_stage_inferred_by_data` | LLM 选的 theme_stage 与量化推断不同 | 必须显式引用 `theme_divergence_reason`，并在 Step 6 评级 COT 里说明是否采纳 LLM 主观判断 |
| `premium_llm_chosen` > `premium_default_template` + 20 | LLM 选的 premium 超默认表 20pp 以上 | 检查 `premium_divergence_reason` 是否有硬数据支撑；无支撑则 Conviction 降一档 |
| 三源 PE 全部缺失（null）| 卖方/历史/同业都无 PE 参照 | data_completeness 必须标注 L3，Conviction 强制"低" |
| `peer_anchor_single_comp` = true | 兄弟股可比仅 1 家（单标的低置信，无第二家纠偏）| Step 7 Conviction **减一档**（估值锚靠单一可比，可靠性打折）|

**核心理念**：不是机械改评级方向，而是按"超共识程度"调 Conviction——机构 PM 看到 "vs consensus +60%" 不会自动 SELL，但会要求 PM 拿出产业证据 defend，否则降仓位（= 降 Conviction）。

⛔ **显式引用要求**（强制留痕，否则视为未应用 Layer 4）：
- 当任一 TRANSPARENCY 字段触发上表中的 Conviction 调档规则时，必须在 **Step 7 评级置信度** 或 **Conviction 校准** 段显式写出一段引用，格式：
  > "TRANSPARENCY.target_pe_high_vs_sell_side_pct = +N%（超共识 N%），无 ≥2 条产业证据 → Conviction -1"
- 即使所有 TRANSPARENCY 字段都在阈值内（不触发调档），也必须显式输出一行：
  > "TRANSPARENCY 检查：所有偏离字段均在阈值内（vs_sell_side=__%, vs_self_p80=__%, vs_peer=__%），Conviction 不调"
- 不允许"隐式应用规则"——下游 harness 审计必须能从 manager.md 文本里 grep 到具体 TRANSPARENCY 字段名

### 0.5 板块对照（由板块对照官 Python 确定性输出，含 fallback 兜底）

{sector_comparison if sector_comparison else "（板块对照缺失，参考默认沪深300 比较）"}

**重点提取**：
- 主题命中？主题 ETF / 主题代表股
- 本股 vs 主题 ETF 的 30d RS（相对强弱）
- 主题内 30d 收益排名（第几 / 共几只）
- 本股 vs 大盘指数（沪深300/科创50/创业板）的 RS

**用法**：
- **Step 6 第三步评级 COT 时强制引用一句**：例如 "板块 RS 30d +X% 跑赢大盘 / 主题内排名第 N / 板块β 主导"
- **作为评级的"反方力量"**：估值偏高时，如果板块 RS 强 + 主题内排名靠前 → Conviction 降一档但不机械改评级；
  反之估值偏高 + 板块 RS 弱 + 主题内排名靠后 → 强化 SELL 信号
- **fallback 路径透明**：报告头部已经显示了"层级 1→2→3→4"的匹配链路，根据匹配级别判定信号可靠度

### 0.6 资金流综合状态（由 Capital Flow Officer Python 确定性输出，无 LLM）

{capital_flow_yaml if capital_flow_yaml else "（资金流数据缺失——非 A 股或数据源异常）"}

**重点提取**：
- `capital_flow_regime`（强势/分化/恶化/中性/数据不足）
- `capital_flow_score`（0-100，第 7 因子）
- `northbound_5d_direction`（净流入/净流出/平衡/数据停滞）→ **Step 6 工具入参 northbound_flow_5d_direction 必须从这里取**（+1=净流入, 0=平衡/数据停滞, -1=净流出）
- `net_inflow_streak_days`（主力资金连续流入/流出天数）
- `ddx_like_5d_pct_1y`（DDX-like 1 年百分位 0-100）

**用法**：
- Step 6 调 `compute_step6_final_rating` 时，催化路的 `northbound_flow_5d_direction` **必须**从上方 CAPITAL_FLOW YAML 的 `northbound_5d_direction` 映射（净流入→+1, 平衡/数据停滞→None, 净流出→-1），禁止从 news 报告中猜测
- Step 8 风险清单：若 `capital_flow_regime` = 恶化，风险维度增加"资金面恶化风险"
- 最终 Thesis 第三节"最关键论据"：若 regime=强势/恶化，计入论据（附 capital_flow_score 数值）

### 0.2 市场共识快照（由共识识别官提炼）

{consensus_snapshot if consensus_snapshot else "（共识缺失，自行从 sentiment+news 推断方向）"}

**重点提取**：direction / strength / crowded / MARKET_IMPLIED_VALUATION.{{market_expected_eps_2026e, market_implied_pe_range, sell_side_target_price_range, industry_pe_median}}

---

## Step 1: 行业框架（Industry Framework）

写一段 100-150 字的行业框架分析，必须显式回答以下 4 个维度：

| 维度 | 输出 |
|------|------|
| **1.1 行业生命周期** | 导入期 / 成长期 / 成熟期 / 衰退期（引用 fundamentals/news 数据） |
| **1.2 关键驱动因子** | 列出 ≤3 个对该行业最关键的驱动因子（如半导体：周期+技术节点+下游需求；消费：渠道+品牌+客单价） |
| **1.3 该行业典型估值范式** | 该行业一般用什么估值方法（如周期股看 PB、成长股看 PEG、消费看 PE/DCF）—— 优先采用 stock_profile.VALUATION_METHOD 的推荐 |
| **1.4 当前行业景气度** | 上行 / 下行 / 底部反转 / 顶部回落（引用 news 报告中的行业新闻支撑） |

**输出格式**：

> Step 1 行业框架：[文字段落]
> 关键结论：
> - 行业阶段：__
> - 关键驱动：1) __ 2) __ 3) __
> - 主估值范式：__（依据：__）
> - 当前景气度：__（依据：__）

---

## Step 2: 公司在行业中的定位

| 维度 | 输出 |
|------|------|
| **2.1 市场地位** | 龙头 / 跟随者 / 挑战者 / 弱势（引用市占率/营收规模数据） |
| **2.2 护城河维度评分**（1-5 分）| 成本：__ / 规模：__ / 品牌：__ / 技术：__ / 网络效应：__ |
| **2.3 竞争格局变化方向** | 公司份额在扩大 / 稳定 / 收缩，依据是什么 |
| **2.4 是否值得相对行业给溢价/折价** | 溢价/折价/无差异 + 一句话说明 |

**输出格式**：

> Step 2 公司定位：[文字段落]
> 关键结论：
> - 市场地位：__
> - 护城河强项：__（评分最高的 1-2 个维度）
> - 竞争格局：__
> - 溢价/折价倾向：__

---

## Step 3: 业绩拐点判断（核心！）

**业绩拐点判断比估值绝对值更重要**。估值告诉你"现在多少钱"，拐点告诉你"未来去哪"。

| 维度 | 输出 |
|------|------|
| **3.1 当前业绩周期阶段** | 业绩刚启动 / 加速期 / 顶部 / 衰退 / 拐点期（引用季度业绩对比） |
| **3.2 季度边际变化方向** | 同比方向 + 环比方向（具体百分比变化）|
| **3.3 业绩可持续性** | 持续 / 一次性 / 待验证（依据：fcf_quality / 单季 vs 年度 EPS 对比 / 是否有大额非经常性损益） |
| **3.4 下一关键检验点** | 何时（具体日期）通过何种数据（如 Q2 营收 / CXL 量产公告）能验证拐点 |

**输出格式**：

> Step 3 业绩拐点：[文字段落]
> 关键结论：
> - 当前周期阶段：__
> - 边际方向：同比 __ / 环比 __
> - 可持续性：__（依据：__）
> - 下一验证点：__（日期/数据）

⚠️ **禁止**只给"基本面向好/恶化"这种定性表述，必须给具体的边际变化数据。

---

## Step 4: 多元估值交叉验证（至少 3 种方法）

按 Step 1.3 识别的估值范式 + stock_profile.VALUATION_METHOD 推荐，**至少用 3 种方法独立计算目标价**：

### 方法清单与适用条件

| 方法 | 适用 | 公式 |
|------|------|------|
| DCF | 现金流稳定的成熟期公司 | 三阶段 DCF 或 简化 PV |
| PE × EPS | 大多数盈利公司 | `target_pe_range`（TTM 倍数）**× EPS_TTM** |
| PEG | 高增长公司 | **必须调 `compute_peg_target_price`**：① `eps`/`growth_rate_pct` 入参直读 `SYS_FORWARD_EPS`/`SYS_PEG_GROWTH_PCT`（确定性，已半衰封顶+低基数护栏，**严禁**自选；同股不同跑选 50%↔33% 是目标价 320↔180 摆动的根）；② ⛔ **PEG 腿目标价必须原样采用工具返回的 `low`/`mid`/`high`，严禁手算、严禁自行设隐含 PE**（隐含 PE = `PEG区间 × growth_rate_pct`，由工具算，不是你另选一个"觉得合理"的高 PE）、**严禁"觉得公式目标太低就上调"**——天孚式同股 PEG 目标 82↔410 摆动，就是 LLM 手算时一次用 PE≈25(公式对)、一次用 PE≈125(自造)乱跳造成的。仅当画像无 SYS_PEG 行时才退回自估 |
| PB × BPS | 重资产/金融/周期 | 目标 PB × 当期 BPS |
| EV/EBITDA | 资本密集型 / 跨周期对比 | 目标倍数 × EBITDA |
| 历史估值分位 | 任何有历史数据的标的 | 当前估值 vs 自身历史 1Y/3Y/5Y 分位 |
| 同业可比 | 行业内有可比标的 | 同业 **TTM** 中位 PE **× EPS_TTM**（TTM×TTM 同口径）|

⚠️ **PE/EPS 口径一致铁律（强制，stock_profile 已给 EPS_TTM 值，直接用）**：
- `target_pe_range` / 同业中位 / `PE_TTM×0.x` **全是 TTM 倍数** → `PE×EPS` 腿 和 `同业可比` 腿 **必须乘 EPS_TTM**。
- **只有 PEG 腿用前瞻 EPS**（前瞻增长由 PEG 单独体现）。
- ❌ **严禁"全部方法统一用前瞻/2026E EPS"**——那会让 TTM 同业中位 × 前瞻 EPS **双重计入成长**，目标价虚高近 50%，把高估值股（如 PE 120 的澜起）错抬成强买。这是历史错评 bug，不许再犯。

### 方法选择规则（强制按 stock_profile.VALUATION_METHOD 选，不再凭感觉）

⚠️ **核心约束（必读）**：方法**不能**自由选 3 种凑数，**必须**按 stock_profile.VALUATION_METHOD 输出的 `primary_method` + `secondary_methods` 来选。这是历史 bug 来源——RM 习惯性给银行股套 PEG、给周期股套 PE×EPS，结构性偏差。

#### 强制选择规则

| 字段 | 用作 | 默认权重 |
|------|------|---------|
| `primary_method`（1 个）| Step 4 方法 1 | **50%** |
| `secondary_methods`（通常 2 个）| Step 4 方法 2 / 方法 3 | **每个 25%** |
| 若 secondary 只有 1 个 | Step 4 方法 2 | 50% |
| 若 secondary 有 3+ 个 | 选最适合的 2 个作为方法 2/3 | 每个 25% |

##### ⛔ 成长股前瞻口径路由（最高优先级，覆盖上表默认）

stock_profile 末尾若有 `SYS_TARGET_PRIMARY_METHOD:` 行（Python 确定性注入），**必须照它执行**，不许用 VALUATION_METHOD 正文里的 primary_method 改判：
- `SYS_TARGET_PRIMARY_METHOD: peg` → **方法 1 必须是 PEG（前瞻 EPS，权重 50%）**；TTM 口径的 `PE×EPS` 腿权重**压到 ≤20%**（它是后视镜，不主导成长龙头 12 个月目标）；剩余权重给卖方目标价/同业可比腿。
- 依据见 `SYS_TARGET_PRIMARY_REASON`；典型权重 = PEG 50% / 卖方目标价或同业 30% / PE×EPS(TTM) 20%。
- **口径不变铁律仍生效**：PEG 腿用前瞻 EPS、PE×EPS 与同业腿仍乘 EPS_TTM——本路由只调权重，不混口径（别又犯"TTM 倍数×前瞻 EPS"双重计入）。
- 为什么：TTM PE×EPS 给快速成长龙头当主腿会系统性压低目标价 15-25%（中际旭创式：base-case 隐含负收益虚高）。对标卖方做法，成长股 12 个月目标本就前瞻。
- 若该行缺失（非成长股/增速不足/discipline）→ 按上面默认表执行。

#### 行业对照（审计 stock_profile.primary_method 是否合理，不覆盖）

银行/保险→pb；周期（钢化煤）→pb/历史分位；SaaS/AI应用/半导体设计/创新药→peg；消费医药→pe_eps；公用事业/高股息→ddm；资源能源→pb；重工业→ev_ebitda；ETF→nav。

若 stock_profile.primary_method 与本对照严重不符（例如银行股给了 peg），在 Step 4 开头先指出该问题，但仍按 stock_profile 原意计算，置信度降一档。

#### 工具覆盖

- ✅ 有专用工具：pe_eps / peg / 同业可比（套行业 PE）/ 历史分位（套自身分位 PE）
- ❌ 手算：pb / dcf / ddm / ev_ebitda / nav —— 公式必须展开（禁止"DCF 估值约 200 元"），关键输入（WACC/g/折现率/EBITDA）显式列出来源；输入只能凭感觉时该方法标"低置信度"，权重砍半

### 输出要求（每种方法独立）

> **方法 1: [方法名]**（来源：stock_profile.primary_method = __）
> - 输入数据：[列出所有用到的数据 + 来源]
> - 计算过程：[公式展开 / 或工具调用]
> - 目标价：__ 元
> - 该方法的局限性：[1 句话]
>
> **方法 2: [方法名]**（来源：stock_profile.secondary_methods[0] = __）
> - ...
>
> **方法 3: [方法名]**（来源：stock_profile.secondary_methods[1] = __）
> - ...

### 综合目标价（权重必须按本节规则选，不再 40/30/30 默认）

> | 方法 | 目标价 | 权重 | 权重来源 |
> |------|--------|------|---------|
> | 方法 1（primary）| __ | 50% | stock_profile.primary_method |
> | 方法 2（secondary[0]）| __ | 25% | stock_profile.secondary_methods |
> | 方法 3（secondary[1]）| __ | 25% | stock_profile.secondary_methods |
>
> **权重例外说明**（如果偏离上表）：__（如某方法标为"低置信度"权重砍半，剩余权重按比例分给其他方法）
>
> **综合目标价区间**：[低位] 至 [高位] 元（**必须调 compute_overlap_target_price 或 compute_weighted_target_price**）

### 重要约束

- **PEG 半衰公式**：预期 EPS = EPS_TTM × (1 + min(增长率, 60%)/2)——防止异常高增长被纯线性外推。**画像已用此公式算好 `SYS_FORWARD_EPS` / `SYS_PEG_GROWTH_PCT`，PEG 腿直读即可，不要自己再算一遍（自算会引入 LLM 漂移）**
- **数据缺失**：如某方法关键输入缺失（如 fundamentals.SUMMARY 的 EPS=null），明确写"该方法不适用，原因：__"，不强行外推
- **禁止凭感觉给目标 PE/PB**：必须引用 stock_profile.target_pe_range / consensus.market_implied_pe_range / 行业可比数据，其中至少一个
- **置信度对照**：综合目标价区间宽度 > 当前价 × 50% → 标"低置信度"
  - ⚠️ **口径例外（成长股前瞻路由）**：若 stock_profile 末尾 `SYS_TARGET_PRIMARY_METHOD: peg`（前瞻路由已激活），则 **PEG(前瞻) 腿 与 PE×EPS(TTM) 腿之间的区间宽度是设计内的"口径差"**（前瞻 vs 后视镜），**不算真实估值分歧/不确定性，不得据此标低置信、不得据此降 Conviction**。此时判低置信只看**前瞻一致的锚之间**的离散度（PEG 腿 与 卖方目标价腿），或扣除 TTM 腿后再算宽度。
  - 理由：是我们**有意**让前瞻 PEG 主导成长股目标价（对标卖方），TTM 腿只作下限参考；为这个有意为之的口径差降信心 = 双重惩罚，会无故压低优质成长票的仓位。
- **时间窗一致性**：forward EPS 不能乘 trailing PE，反之亦然。Step 4 顶部必须显式说明使用的 EPS 是哪个口径（TTM / 2025E / 2026E）+ PE 是哪个口径（TTM / forward），两者必须匹配

---

## Step 5: Bull / Base / Bear 三情景

不是简单的概率加权，而是 RM 对未来三种可能路径的真实判断。

| 情景 | 概率 | 目标价 | 核心假设（可观测） |
|------|------|--------|------------------|
| **Bull case** | __% | __ 元 | <核心假设：什么具体事件兑现> |
| **Base case** | __% | __ 元 | <核心假设：维持当前路径的话怎样> |
| **Bear case** | __% | __ 元 | <核心假设：什么具体风险兑现> |

**约束**：
- 概率加总 = 100%
- Base case 概率必须在 40-60%（默认假设市场延续当前路径）
- **Base case 目标价必须在 Step 4 综合目标价区间内**（Base 代表"维持估值方法的中性路径"）
- Bull case 目标价**通常** ≤ Step 4 上限 × 1.5（超出需充分论证）
- Bear case 目标价**通常** ≥ Step 4 下限 × 0.6（超出需充分论证）
- 核心假设必须可观测可证伪（"业绩超预期"不可接受，"Q2 营收 >25%"可接受）

**Base case 是真实预期，不是中位数**——你认为最可能发生的路径，不是 Bull 和 Bear 的平均。

### ⚠️ Step 5 末尾强制工具调用：一致性检验

设计完 Bull/Base/Bear 三档目标价后，**必须**调用 `compute_scenario_consistency_check` 工具检查三情景与 Step 4 估值区间的一致性，输入：

```
step4_target_low / step4_target_high → 必须用 Step 4 综合目标价区间（来自 compute_overlap_target_price 或 compute_weighted_target_price 工具输出）
bull_target / base_target / bear_target → 你刚设计的三情景目标价
```

**强制规则**：
- 若工具返回 `ok: True`（无 warning）→ Step 5 通过，继续 Step 6
- 若工具返回 `ok: False`（有 warning）→ **必须逐条回应**：
  - Base 超出 Step 4 区间 → **必须修正 Base case 或回头改 Step 4 估值方法**（禁止"加一句解释"就过）
  - Bull/Bear 超出弱阈值 → 在 Step 5 末尾补充论证（具体催化、anchor 失效路径）
- **禁止**自评"约束验证 ✅"——这是工具的工作，不是你写一句话就过

---

## Step 6: 综合评级判断（一次合议，工具执行）

**评级终段是一次合议，不是流水线**。动态阈值 → 估值五档映射 → regime 闸门 → PEG 低置信收敛 → 拥挤度 → 对称升降档 → 趋势叠加 → 极端背离防御 → 全链不变量终检，全部在 `compute_step6_final_rating` **一次工具调用**内按固定顺序执行（历史教训：这九道关卡分步执行且互相不知情时，出过"ride 托底的 HOLD 被趋势叠加再升成 OVERWEIGHT——偏离 +133% 的票评级看多"和"拥挤多头禁 BUY 被叠加绕回 BUY"两类事故）。

你的工作只有三件：① 备齐输入（全部直读，来源见下表）；② 真正发起这一次工具调用；③ 写评级 COT 解释链路。

⛔ **严禁narrate**：必须真正发起 `compute_step6_final_rating` 工具调用并采用其返回的 `final_rating`，禁止手写伪 JSON 自圆其说——伪造的返回一律视为无效，评级作废。
⛔ **无人工微调通道**：旧版"例外情形可 ±1 档"已全部收进工具（业绩拐点 / 数据完整度 / 红旗 / 拥挤度都是工具入参）。若认为评级不合理，唯一合法途径是回头修改 Step 4 估值并重新调工具，禁止在工具输出之外加减档、禁止"短期透支 / 估值消化 / 市场情绪"式的主观改判。

### 第一步：备齐输入（逐项列出，全部直读）

| 入参 | 直读来源 | 说明 |
|------|---------|------|
| current_price | instrument_context / market_report | 当前价 P_0 |
| target_price_mid | **Step 4 的 compute_weighted_target_price 或 compute_overlap_target_price 工具输出的 mid** | 禁止自算调整、禁止用 Base case 反推 |
| target_price_source | 审计字段 | 如『compute_weighted_target_price 工具输出』 |
| style | stock_profile.style | 阈值 style 系数由工具内部查表，你不算 |
| theme_premium_pct | **画像末尾 `SYS_THEME_PREMIUM_PCT:` 行** | Python 确定性值（已按 regime 闸门）；禁止从 theme_stage 重算 |
| theme_stage | THEMATIC_PREMIUM.theme_stage | 仅用于 fading 上沿锁 |
| valuation_regime | **画像末尾 `SYS_VALUATION_REGIME:` 行** | 照抄 ride/neutral/discipline；严禁臆断或从正文猜；全文搜不到才填 "" |
| peg_confidence | 画像末尾 `SYS_PEG_CONFIDENCE:` 行 | normal / low；无此行填 "" |
| consensus_crowded | consensus_snapshot.crowded | true/false（软标志，工具内须经硬数据确认才生效） |
| consensus_direction | consensus_snapshot.direction | 偏多 / 偏空 / 中性 |
| quant_anticrowding | QUANT_SCORE.factor_scores.anticrowding | 0-100；拥挤的硬确认之一（≤30 = 真拥挤），缺失填 None |
| retail_concentration_signal | 0.6 节 CAPITAL_FLOW.retail_concentration_signal | 散户高接盘 / 中性；拥挤的硬确认之二，缺失填 "" |
| inflection_stage | 你 Step 3 的『当前周期阶段』 | 加速期 / 底部反转 / 顶部 / 衰退 / 拐点期 |
| data_completeness | VALUATION_METHOD.data_completeness | L0-L3 |
| red_flags_count | fundamentals.SUMMARY.red_flags 条数 | 整数 |
| earnings_sustainability | 你 Step 3 的『可持续性』 | 持续 / 一次性 / 待验证 |
| bear_anchor_strong | 辅助分析中空头 anchor 论据是否有 hard data 强支撑 | true/false |
| decision_style | stock_profile.DECISION_STYLE | momentum 风格不靠低估升档 |
| composite_score / momentum_score | QUANT_SCORE.composite / factor_scores.momentum | 数值 |
| market_weight / news_weight / sentiment_weight | stock_profile.REPORT_WEIGHTS | 0-1 |
| market/news/sentiment_direction_vote | 你读完三份报告后各给一票 | +1.0 全面看多 / +0.5 偏多有保留 / 0 中性 / -0.5 偏空有保留 / -1.0 全面看空 |
| sell_side_target_change_pct | 近 30 日卖方目标价中位变化% | **缺失填 None，禁止编造** |
| institutional_holding_change_pct | 近 1 季机构持仓变化% | 缺失填 None，禁止编造 |
| northbound_flow_5d_direction | **0.6 节 CAPITAL_FLOW.northbound_5d_direction 映射** | 净流入→1，净流出→-1，平衡/数据停滞→None；禁止从 news/sentiment 猜 |
| kol_bullish_ratio_trend_pct | KOL 多头率相对 30 日均的变化(pp) | 缺失填 None |
| inflection_confirmed_recent | 业绩拐点是否刚被新数据（如刚出的季报）确认 | true 时极端背离防御跳过（量化锚滞后于新数据） |

### 第二步：一次工具调用（强制）

调用 `compute_step6_final_rating`，**最终评级 = 工具返回的 `final_rating`，原样采用**。

工具内部链路（你不执行，只需在 COT 里复述返回的 explanation）：
1. 动态阈值 = 基础 ±15/±35 × style 系数（blue_chip/cyclical/etf 1.0、illiquid 0.7、high_beta_growth 1.5、theme_speculation 2.0）× (1 + SYS_THEME_PREMIUM_PCT/100)；theme_stage=fading 时上沿锁 30%
2. 估值五档映射 + regime 闸门（ride 把 UNDERWEIGHT/SELL 托底 HOLD；discipline 把 OVERWEIGHT/BUY 封顶 HOLD、SELL 保留；neutral 收敛三档）+ PEG 低置信边界收敛
3. 拥挤度（拥挤多头禁 BUY、拥挤空头禁 SELL；**禁令固化为边界，对后续所有步骤持续生效，趋势叠加绕不过**。共识官的 crowded 只是软标志，须经硬数据确认——反拥挤分 ≤30 或 散户高接盘——才触发；无硬确认不动评级，对标投研用持仓/成交数据判拥挤而非舆情观感）
4. 对称升降档（升档需同时满足：拐点加速/底部反转 + L0/L1 + 红旗≤1 + 低估区 + 非 momentum，且仅 HOLD→OW / OW→BUY；降档 L3 / 红旗≥3 / 拐点顶部衰退 / 空头anchor强+待验证 各 -1，合计最多 -2）
5. 趋势叠加三路（style 动量 / 报告加权方向票 / 催化硬数据，合成最多 ±1）
6. 极端背离防御（composite ≤20 压看多 → HOLD；≥80 托看空 → HOLD）
7. **两条全链不变量**：
   - **闸门边界不可被下游反转**：ride 托底产生的 HOLD 设地板、discipline 封顶产生的 HOLD 设天花板，后续任何升降档不得越过
   - **评级方向必须与隐含收益同号**：最终看多（BUY/OVERWEIGHT）要求目标价中位 > 现价；看空（UNDERWEIGHT/SELL）要求 < 现价；违反者收敛 HOLD——真实投研不存在『看多但目标价在现价下方』的票

#### 输出格式（强制）

> 工具调用：`compute_step6_final_rating`
> 关键输入：current_price=__, target_price_mid=__（来源：__）, style=__, SYS_THEME_PREMIUM_PCT=__, SYS_VALUATION_REGIME=__, peg_confidence=__, crowded=__/方向__, 拐点=__, 数据完整度=__, 红旗=__, composite=__, momentum=__, 三票=__/__/__
> 工具返回：阈值 ±__/±__ | 偏离 __% | rating_raw=__ → regime 闸门 __ → 拥挤 __ → 升降档 __ → 趋势叠加 __（style __ / vote __ / catalyst __）→ 极端防御 __ → 不变量终检 → **final_rating = __**
> 边界与不变量：__（照抄工具返回 bounds.sources 与 stages.e_sign_invariant.note）

### 第三步：评级 COT（200-300 字）

写一段 200-300 字的『评级 COT』，解释工具链路的判断为什么成立，必须包含：

1. **当前价 vs 综合目标价区间**的位置（结合动态阈值，区间下沿/区间内/区间上沿/超出区间）
2. **业绩拐点判断**对评级的支撑/反对
3. **Base case 目标价**是否对当前价格有吸引力
4. **多元估值交叉一致性**（3 种方法是否一致指向同一方向）
5. **如果偏离行业典型估值范式**，说明为什么这么做
6. **板块相对强弱（强制引用一句话）**：从 0.5 节 sector_comparison 取出本股 vs 主题 ETF 或大盘指数的 30d RS，
   作为评级的支撑/反方力量。例：
   - 『板块 RS 30d +12%，主题内排名第 2/5，板块β 主导，估值偏高但短期支撑评级不至于 SELL』
   - 『板块 RS 30d -5%，主题内倒数第 3，板块走弱+估值偏高 → 强化 UNDERWEIGHT 信号』
7. **反方推理**：明确回答『为什么不给更激进/更保守的评级』；若工具链路中闸门/不变量改写了评级（rating_raw ≠ final_rating），逐段复述工具 explanation 说明每次改写的依据

---

## Step 7: 评级置信度

| 等级 | 触发条件 |
|------|---------|
| **高** | 3 种估值方法目标价相对偏离 < 15% + 业绩拐点明确（数据指向同一方向） + 数据完整度 L0/L1 |
| **中** | 3 种估值方法相对偏离 15-30% / 拐点判断有不确定性 / 数据完整度 L2 |
| **低** | 估值方法分歧大（> 30%）/ 拐点完全不明 / 数据完整度 L3 / 多空双方都有强证据但方向不明 |

⛔ **SYS_PEG_CONFIDENCE 闸（确定性，强制）**：画像末尾若 `SYS_PEG_CONFIDENCE: low`（单季增速远高于年度=低基数尖峰，前瞻 EPS 高度不确定），则：
- 评级置信度**最高只能给"中"**（不许"高"），并在理由里点名"前瞻增速含低基数尖峰、PEG 锚不确定"；
- **且偏离度落在阈值边界 ±5pp 内时（如 UNDERWEIGHT/OVERWEIGHT 区刚过下/上沿），优先收敛到 HOLD**——数据本就说不清方向，不下强方向单（对标真实投研：预测分歧极大时给中性、低仓位，不反复横跳 OW↔UW）。

输出：**评级置信度：[高/中/低]，理由：__**

---

## Step 8: 风险清单（不修改评级方向，给 PM 仓位参考）

| 风险维度 | 是否存在 | 严重程度 | 影响 PM 哪个决策 |
|---------|---------|---------|----------------|
| 拥挤交易（multi-head/short） | yes/no | 高/中/低 | 仓位上限 |
| 单一催化剂依赖 | yes/no | 高/中/低 | 时间止损节奏 |
| 数据完整度低 | yes/no | 高/中/低 | Conviction 调整 |
| 流动性差 | yes/no | 高/中/低 | 分批节奏 |
| 一次性损益占比高 | yes/no | 高/中/低 | Conviction 调整 |
| 强 anchor 失效风险（如某催化推迟）| yes/no | 高/中/低 | Time Stop 触发条件 |

---

# 辅助分析：多空辩论评分 → Conviction 校准（不影响评级方向）

辩论分析不主导评级，只用于 Conviction 校准。

## 反驳质量评估（评分前必做）

- **多头被有效反驳的论据**（最多 2 条）：哪条多头论据被空头用数据驳倒？驳倒方式（引用错误数据 / 提出反例数据 / 揭示前提失效）
- **空头被有效反驳的论据**（最多 2 条）：同上

**反驳影响打分**：被有效反驳的论据 -1 ~ -2 分；引用错误数据的论据直接 0 分剔除。

## 多空辩论评分（保留三因子加权）

权重 = 证据类型 × Hard Data × 画像权重

| 证据类型 | 基础权重 |
|---------|---------|
| Hard fact | 3.0 |
| Catalyst | 2.0 |
| 估值类比/趋势外推 | 1.0 |
| 情绪/叙事 | 0.5 |

Hard Data 修正：yes 不变；no × 0.5
画像权重修正：来源报告权重 ≥35% × 1.2；20-35% × 1.0；<20% × 0.8
共识对称校准：反共识 +1（无论原始得分）；共识封顶 6

### 多头论据明细

| # | 论据 | 立场 | 来源 | 证据类型 | Hard Data | 原始得分 | 共识校准 | 反驳调整 | 最终得分 | 最终权重 |
|---|------|------|------|---------|-----------|---------|---------|---------|---------|---------|
| 1 | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

**Bull Score = Σ(最终得分 × 最终权重) / Σ权重 = __**

### 空头论据明细

| # | 论据 | 立场 | 来源 | 证据类型 | Hard Data | 原始得分 | 共识校准 | 反驳调整 | 最终得分 | 最终权重 |
|---|------|------|------|---------|-----------|---------|---------|---------|---------|---------|
| 1 | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

**Bear Score = __**

**d = Bull Score - Bear Score = __**

## Conviction 校准规则

基于上述评分 d，对 Step 7 的评级置信度做最终校准：

| d 范围 | Conviction 校准 |
|-------|----------------|
| d > 1.5 | Conviction +1 档（最多到"高"）|
| 1.0 < d ≤ 1.5 | Conviction 不变 |
| -1.0 < d ≤ 1.0 | Conviction 不变 |
| -1.5 ≤ d < -1.0 | Conviction 不变 |
| d < -1.5 | Conviction -1 档（最低到"低"）|

**额外约束**：
- 若反驳质量评估中有"多头侧 anchor 被有效反驳"且原评级是偏多 → Conviction -1
- 若反驳质量评估中有"空头侧 anchor 被有效反驳"且原评级是偏空 → Conviction -1

**最终 Conviction**：[高/中/低]，校准过程：__

---

# 最终输出：Thesis 报告

按以下结构输出最终 thesis：

## 一、评级与置信度

| 字段 | 内容 |
|------|------|
| 最终评级 R | __ |
| Conviction | __（理由：__）|
| 综合目标价区间 | __ 至 __ 元 |
| 当前价 | __ 元 |
| Base case 目标价 | __ 元（概率 __%）|
| 隐含 Base case 收益 | __ % |

## 二、核心 Thesis（200 字内）

复述 Step 6 的评级 COT，但更精炼。引用 Step 4 估值交叉 + Step 3 拐点判断 + Step 5 三情景为主线。

## 三、3-5 条最关键论据

来自 Step 4-5 + 多空辩论的最强论据（hard fact 优先）。每条 1-2 句话，注明立场（共识/反共识）。

## 四、证伪触发器（3 条）

每条：触发条件（可观测）+ 时间窗口 + 触发动作（评级降档至 X）。

## 五、反面风险（一句话）

当前 thesis 最大反面风险，若错最可能错在哪里。

## 六、风控审查指引

把 Step 8 的风险清单转化为风控团队的审查重点。

---

## 原始分析师报告（交叉校验数据用）

[置信度:高] Company fundamentals report: {fundamentals_report}

[置信度:中高] Market research report: {market_research_report}

[置信度:中] Latest world affairs news: {news_report}

[置信度:中低] Social media sentiment report: {sentiment_report}

## 多空辩论记录（仅用于辅助 Conviction 校准）

{history}

---

{RISK_DEBATE_PHRASING_RULES}

**重要：请用中文撰写你的 thesis 报告。** 评级关键词（Buy/Overweight/Hold/Underweight/Sell）和股票代码请保留英文原文。

**最后提醒**：
- 你的评级是 8 步 COT 综合判断的产出，**不是 d 阈值的公式输出**
- 多空辩论评分只影响 Conviction，不影响评级方向
- 你只出 thesis，不出执行细节（建仓价/止损/仓位由 PM 决定）

---

## ⚠️ 报告末尾强制输出 RM_SUMMARY YAML（用于 harness 自动归档）

报告**完成后**，必须在最末尾输出一段 YAML 摘要，**字段名严格按以下格式**，否则归档失败。
所有数值直接采用 8 步 COT 工具调用得到的结果，不要再调整。

```yaml
RM_SUMMARY:
  ticker: "{rm_ticker}"                  # 已填好，请勿修改
  trade_date: "{rm_trade_date}"          # 已填好，请勿修改
  current_price: <float>                 # 当前价 P_0（与 Step 6 使用的一致）
  rm_rating: BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL
  rm_conviction: 高 / 中 / 低
  target_price_low: <float>              # Step 4 综合目标价区间下沿
  target_price_mid: <float>              # Step 4 工具返回的中位数
  target_price_high: <float>             # Step 4 综合目标价区间上沿
  bull_target: <float>                   # Step 5 三情景
  bull_prob: <float>                     # 0-1 之间（如 0.25）
  base_target: <float>
  base_prob: <float>
  bear_target: <float>
  bear_prob: <float>
  base_case_expected_return_pct: <float> # Base case 隐含收益百分比
  style: <stock_profile.style>           # 已确定字段
  theme_stage: peak / acceleration / initiation / fading / none
  composite_score: <float>               # QUANT_SCORE.composite
  momentum_score: <float>                # QUANT_SCORE.factor_scores.momentum
  deviation_pct: <float>                 # compute_step6_final_rating 返回的 deviation_pct
  threshold_dn_pct: <float>              # 工具返回的动态阈值下沿
  threshold_up_pct: <float>              # 工具返回的动态阈值上沿
  valuation_regime: ride / neutral / discipline   # 画像 SYS_VALUATION_REGIME 原值（回测分腿归因用）
  regime_legs: "<照抄画像 SYS_VALUATION_REGIME_LEGS 行冒号后的内容，整体加双引号>"
  rating_raw: BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL  # 工具返回 rating_raw（估值原始倾向，未过闸门）
  peg_confidence: normal / low / null    # 画像 SYS_PEG_CONFIDENCE（无该行填 null）
  overlay_style_adj: <int>               # 工具返回 overlay_components.style（-1/0/+1）
  overlay_vote_adj: <int>                # 工具返回 overlay_components.vote
  overlay_catalyst_adj: <int>            # 工具返回 overlay_components.catalyst
```

**约束**：
- 缺数据填 `null`（如某只股 momentum_score 缺失）
- 不要嵌套、不要加注释行；本节是供 Python 解析的固定格式
- 该 YAML 必须是报告最后一段，前后用 `---` 分隔，方便提取器定位
"""
        # 绑定 RM 计算工具，让 LLM 在 Step 4 / 辅助分析等数值步骤显式调用工具
        # 替代之前 LLM 心算导致的 Bull Score / 目标价区间 等计算 bug
        llm_with_tools = llm.bind_tools(RM_TOOLS)
        response = _run_tool_calling_loop(llm_with_tools, [HumanMessage(content=prompt)])

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
