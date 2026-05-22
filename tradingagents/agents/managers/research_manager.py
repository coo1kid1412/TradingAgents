import logging
import json

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from tradingagents.agents.utils.agent_utils import build_instrument_context, RISK_DEBATE_PHRASING_RULES
from tradingagents.agents.managers.rm_tools import RM_TOOLS, RM_TOOLS_BY_NAME

logger = logging.getLogger(__name__)

# 工具调用循环上限——避免 LLM 反复调同一工具陷入死循环
# 15 = 9 基础工具 + 6 Step 6 趋势叠加链路（机械映射 / 一致性 / style / vote / catalyst / synthesis）+ 缓冲
_MAX_TOOL_ITERATIONS = 15


def _run_tool_calling_loop(llm_with_tools, initial_messages):
    """执行 LLM 工具调用循环，直到 LLM 不再调工具或达到上限。

    返回 AIMessage，其 content 是**所有迭代的 LLM 文本累积**——保留完整 8 步 COT 链路。
    历史 bug：之前只返回最后一次 response.content，导致 Step 1-8 + 第六步 COT 全丢，
    judge_decision 只剩"最终输出 Thesis 报告"段。
    """
    messages = list(initial_messages)
    cot_segments: list[str] = []

    for iteration in range(_MAX_TOOL_ITERATIONS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # 累积本轮 LLM 输出的文本（即使含 tool_calls，也常含 Step COT 片段）
        content = (response.content or "").strip()
        if content:
            cot_segments.append(content)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            logger.info(
                "RM tool calling 循环结束（第 %d 轮，累积 %d 段 COT，总长 %d 字符）",
                iteration + 1, len(cot_segments), sum(len(s) for s in cot_segments),
            )
            return AIMessage(content="\n\n".join(cot_segments))

        logger.info("RM 第 %d 轮工具调用：%d 个工具", iteration + 1, len(tool_calls))
        for tc in tool_calls:
            tool_name = tc.get("name")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id", "")

            tool = RM_TOOLS_BY_NAME.get(tool_name)
            if tool is None:
                error_msg = f"未知工具：{tool_name}"
                logger.warning(error_msg)
                messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
                continue

            try:
                result = tool.invoke(tool_args)
                result_str = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
                logger.debug("RM 工具 %s 结果: %s", tool_name, result_str[:200])
                messages.append(ToolMessage(content=result_str, tool_call_id=tool_id))
            except Exception as e:
                error_msg = f"工具 {tool_name} 执行失败: {e}"
                logger.warning(error_msg)
                messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))

    # 达到上限仍未结束——强制再调一次，要求 LLM 不再用工具直接给最终答案
    logger.warning("RM 达到工具调用上限 (%d 轮)，强制 LLM 续写最终结论", _MAX_TOOL_ITERATIONS)
    messages.append(HumanMessage(
        content="你已经调用足够多次工具了。请基于已有的工具结果直接写出最终的 thesis 报告，"
                "**不要再调用任何工具**。"
    ))
    final = llm_with_tools.invoke(messages)
    final_content = (final.content or "").strip()
    if final_content:
        cot_segments.append(final_content)
    return AIMessage(content="\n\n".join(cot_segments))


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

## ⚠️ 数值计算必须调用工具（强制约束）

你已经绑定了 15 个计算工具。**所有数值计算必须通过工具调用完成，禁止在报告里"心算"或"凭感觉"算**。

| 计算场景 | 必须调用的工具 |
|---------|-------------|
| Step 4 估值方法 1 PE×EPS | `compute_pe_eps_target_price` |
| Step 4 估值方法 PEG | `compute_peg_target_price` |
| Step 4 综合目标价区间（严格重叠）| `compute_overlap_target_price` |
| Step 4 综合目标价（加权折中，当无严格重叠时）| `compute_weighted_target_price` |
| **Step 5 三情景 vs Step 4 一致性检验**（**强制**） | `compute_scenario_consistency_check` |
| Step 5 三情景概率加权 E | `compute_scenario_weighted_e` |
| **Step 6 第三步评级机械映射**（**强制**）| `compute_step6_rating_mapping` |
| **Step 6 第六步.1 Style 趋势调整**（**强制**）| `compute_step6_style_adjustment` |
| **Step 6 第六步.2 非估值方向票调整**（**强制**）| `compute_step6_report_weighted_vote_adjustment` |
| **Step 6 第六步.3 催化动量调整**（**强制**，缺数据可降级）| `compute_step6_catalyst_momentum_adjustment` |
| **Step 6 第六步.4 三类信号合成**（**强制**）| `compute_step6_adjustment_synthesis` |
| 多空辩论 Bull Score | `compute_bull_bear_score`（传 Bull 论据列表）|
| 多空辩论 Bear Score | `compute_bull_bear_score`（传 Bear 论据列表）|
| 得分差 d | `compute_score_difference` |
| Conviction 校准 | `compute_conviction_calibration` |
| 赔率 R 和单一胜率 E（简化兜底）| `compute_odds_and_expected_return` |

**规则**：
- 凡涉及加权平均、目标价区间运算、概率加权、PEG/PE×EPS 公式的地方，**必须**调用工具
- 工具返回结果后，**直接采用工具的数值**，不要再"调整"或"修正"
- 报告中引用数值时显式说明"工具计算结果：__"（透明可追溯）
- ⛔ **禁止**：自己心算后写入报告（这是过往 bug 来源）

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

{quant_score if quant_score else "（量化锚缺失，Step 6 第六步 / 第七步跳过）"}

**重点提取**：
- QUANT_SCORE.composite（0-100 综合分）
- QUANT_SCORE.factor_scores.momentum（动量子分，0-100）
- 其他 5 类因子分项

**用法**：
- 这是**独立于 LLM 判断的量化锚**，由 Python 直接基于动量/价值/质量/成长/低波/反拥挤 6 因子打分得出
- **Step 6 第六步（Style-Conditional 趋势叠加）**会强制把 composite + momentum 喂给 `compute_step6_style_adjustment` 工具，按 style 差异化调整评级（最多 ±1 档，不跨 HOLD）
- **Step 6 第七步（极端背离防御）**作为兜底——只在 composite ≤20 或 ≥80 这种极端值时触发强制调整
- factor_scores 中 <30 分的单项代表该维度有显著风险，Step 8 风险清单应当独立列出

**重点提取**：style / industry / VALUATION_METHOD.primary_method / target_pe_range / target_pb_range / data_completeness

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
| PE × EPS | 大多数盈利公司 | 目标 PE × 预期 EPS |
| PEG | 高增长公司 | 目标 PEG × 增长率 × 预期 EPS |
| PB × BPS | 重资产/金融/周期 | 目标 PB × 当期 BPS |
| EV/EBITDA | 资本密集型 / 跨周期对比 | 目标倍数 × EBITDA |
| 历史估值分位 | 任何有历史数据的标的 | 当前估值 vs 自身历史 1Y/3Y/5Y 分位 |
| 同业可比 | 行业内有可比标的 | 同业平均 PE/PB × 公司财务指标 |

### 方法选择规则（强制按 stock_profile.VALUATION_METHOD 选，不再凭感觉）

⚠️ **核心约束（必读）**：方法**不能**自由选 3 种凑数，**必须**按 stock_profile.VALUATION_METHOD 输出的 `primary_method` + `secondary_methods` 来选。这是历史 bug 来源——RM 习惯性给银行股套 PEG、给周期股套 PE×EPS，结构性偏差。

#### 强制选择规则

| 字段 | 用作 | 默认权重 |
|------|------|---------|
| `primary_method`（1 个）| Step 4 方法 1 | **50%** |
| `secondary_methods`（通常 2 个）| Step 4 方法 2 / 方法 3 | **每个 25%** |
| 若 secondary 只有 1 个 | Step 4 方法 2 | 50% |
| 若 secondary 有 3+ 个 | 选最适合的 2 个作为方法 2/3 | 每个 25% |

#### 行业对照（用于审计 stock_profile 是否合理，不是用来覆盖它）

| 典型行业 | 应该看到的 primary | 应该看到的 secondaries |
|---------|------------------|---------------------|
| 银行/保险 | pb | 历史分位、DDM、同业可比 |
| 钢铁/化工/煤炭（周期）| pb / 历史分位 | 同业可比、EV/EBITDA |
| 高成长 SaaS / AI 应用 | peg | PE×EPS、同业可比 |
| 半导体设计 / 创新药 | peg | PE×EPS、同业可比 |
| 消费 / 家电 / 医药 | pe_eps | 同业可比、DCF |
| 公用事业 / 高股息蓝筹 | ddm / 股息率倒推 | PE×EPS、DCF |
| 资源 / 能源 | pb | EV/EBITDA、PE×EPS |
| 资本密集型重工业 | ev_ebitda | PB×BPS、PE×EPS |
| ETF / 封闭式基金 | nav / 折溢价率 | （单一方法即可）|

**若 stock_profile.primary_method 与本表严重不符**（例如银行股给了 peg）：
- 在 Step 4 开头**先指出该问题**
- 仍按 stock_profile 原意计算（不要私自换方法），但在置信度评估时降一档（"估值方法选择存疑"）
- 不要私自把方法换掉——这是 stock_profile 的责任

#### 已实现的工具覆盖矩阵（决定能否用 compute_*_target_price 工具）

| primary_method | 是否有专用工具 | 计算方式 |
|---------------|-------------|---------|
| pe_eps | ✅ compute_pe_eps_target_price | 工具计算 |
| peg | ✅ compute_peg_target_price | 工具计算 |
| pb | ❌ 无专用工具 | LLM 手算：目标 PB × 当期 BPS（必须显式展示公式） |
| dcf | ❌ 无专用工具 | LLM 手算：三阶段 DCF，须显式给出 WACC / 永续 g / 现金流预测 |
| ddm | ❌ 无专用工具 | LLM 手算：DDM = 每股股息 / (折现率 − 增长率) |
| ev_ebitda | ❌ 无专用工具 | LLM 手算：目标倍数 × EBITDA（市值 = EV − 净债务） |
| nav | ❌ 无专用工具 | LLM 手算（ETF/封基用） |
| 同业可比 | 通过 compute_pe_eps_target_price 套行业 PE 实现 | 目标 PE = 行业 PE 中位数 ± 龙头/折价调整 |
| 历史分位 | 通过 compute_pe_eps_target_price 套自身历史 PE 实现 | 目标 PE = 自身历史 PE 分位（25%/50%/75%）|

⚠️ **手算时强制要求**：
- 公式必须展开成可审计形式（不能写"DCF 估值约 200 元"）
- 关键输入（WACC / g / 折现率 / EBITDA 等）必须**显式列出**，说明来源
- 若关键输入只能凭感觉给（例如 WACC 没有行业基准）→ 该方法标为"低置信度"，权重砍半（primary 50% → 25%；secondary 25% → 12.5%）

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

- **PEG 半衰公式**：预期 EPS = EPS_TTM × (1 + min(增长率, 60%)/2)——防止异常高增长被纯线性外推
- **数据缺失**：如某方法关键输入缺失（如 fundamentals.SUMMARY 的 EPS=null），明确写"该方法不适用，原因：__"，不强行外推
- **禁止凭感觉给目标 PE/PB**：必须引用 stock_profile.target_pe_range / consensus.market_implied_pe_range / 行业可比数据，其中至少一个
- **置信度对照**：综合目标价区间宽度 > 当前价 × 50% → 标"低置信度"
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

## Step 6: 综合评级判断（COT 主观）

**评级不是公式产出，是 RM 基于 Step 1-5 综合判断后的主观决定**。

### 第一步：计算动态估值偏离阈值（强制，先算阈值再判评级）

**核心理念**：不同股性 + 主题热度对估值偏离的容忍度天然不同。蓝筹股 PE 偏离 +15% 就该警惕；AI 算力加速期的龙头 PE 偏离 +80% 仍可能合理。**用一刀切的 ±10%/±30% 会错杀整轮主题上升周期**（参考美股：NVDA 2024 PE 偏离历史中位 +240% 仍维持 BUY，TSLA 2020 PE 偏离 +500% 仍 BUY）。

#### 阈值计算公式

```
最终偏离阈值 = 基础阈值 × style 系数 × (1 + theme_premium_pct / 100)

基础阈值：±15% / ±35%（默认温和宽容）

style 系数（来自 stock_profile.style）：
  blue_chip          × 1.0    （蓝筹守严格）
  cyclical           × 1.0    （周期股看 PB 不看 PE 偏离）
  illiquid           × 0.7    （流动性差更严格保护）
  etf                × 1.0
  high_beta_growth   × 1.5    （成长股容忍偏离）
  theme_speculation  × 2.0    （题材炒作大幅放宽）

theme_premium_pct（来自 stock_profile.THEMATIC_PREMIUM）：
  启动期 initiation   +30%
  加速期 acceleration +50%    ← 主题最宽容
  顶部期 peak         +20%    （开始警惕）
  退潮期 fading       -20%    （主题反噬，反向收紧）
  不在主题 none       +0%
```

#### 计算示例

| 标的 | style | theme_stage | 阈值计算 | UNDERWEIGHT 触发偏离 |
|------|-------|-------------|---------|---------------------|
| 寒武纪（AI 算力加速期）| high_beta_growth | acceleration | 35 × 1.5 × 1.5 | **+78.75%** |
| 中际旭创（CPO 加速期）| high_beta_growth | acceleration | 35 × 1.5 × 1.5 | **+78.75%** |
| 招商银行（不在主题）| blue_chip | none | 35 × 1.0 × 1.0 | **+35%** |
| 某算力租赁小盘 | theme_speculation | acceleration | 35 × 2.0 × 1.5 | **+105%** |
| 某退潮主题股 | theme_speculation | fading | 35 × 2.0 × 0.8 | **+56%**（已收紧）|

#### 评级阈值表（按动态阈值映射）

```
设动态偏离阈值（下沿 −threshold_dn%，上沿 +threshold_up%）：
  - threshold_dn = base_15 × style × theme_factor
  - threshold_up = base_35 × style × theme_factor

  偏离 < -threshold_up  → BUY（深度低估）
  -threshold_up ~ -threshold_dn → OVERWEIGHT（低估）
  ±threshold_dn 区间   → HOLD（合理估值）
  +threshold_dn ~ +threshold_up → UNDERWEIGHT
  > +threshold_up      → SELL（明显高估）
```

#### ⚠️ 主题溢价上限（硬保护，激进版）

即使主题最热也不能无限放宽：

```
任何情况下，最终阈值不能超过：
- BUY 评级最大偏离容忍：+100%（即超过 +100% 强制降至 OVERWEIGHT）
- OVERWEIGHT 最大偏离容忍：+150%（超过强制降至 HOLD）
- 主题退潮期：直接锁定上限到 +30%（主题反噬保护，不再放宽）
```

#### 强制输出（在评级 COT 前显式列出）

```
- stock_profile.style = __
- THEMATIC_PREMIUM.theme_stage = __  / theme_name = __
- 动态阈值计算：基础 35 × style __ × theme __ = ±__ / ±__
- 当前偏离度 = (当前价 - 综合目标价中位) / 综合目标价中位 = ±__%
- 触发评级方向：BUY/OVERWEIGHT/HOLD/UNDERWEIGHT/SELL
- 是否触发硬上限保护：是/否（含原因）
```

---

### 第二步：综合评级 COT（200-300 字）

写一段 200-300 字的"评级 COT"，必须包含以下要素：

1. **当前价 vs 综合目标价区间**的位置（结合动态阈值，区间下沿/区间内/区间上沿/超出区间）
2. **业绩拐点判断**对评级的支撑/反对
3. **Base case 目标价**是否对当前价格有吸引力
4. **多元估值交叉一致性**（3 种方法是否一致指向同一方向）
5. **如果偏离行业典型估值范式**，说明为什么这么做
6. **反方推理**：明确回答"为什么不给更激进/更保守的评级"

### 第三步：评级（5 档之一）— **强制调用 `compute_step6_rating_mapping` 工具**

⚠️ **核心约束（必读）**：评级映射这一步**完全交给 Python 工具**，你只负责：
1. 提供正确输入（当前价 + Step 4 目标价中位 + 动态阈值）
2. 读取工具返回的评级
3. 写评级 COT 解释（200-300 字）

#### 强制工具调用（必须）

**必须**调用 `compute_step6_rating_mapping` 工具，输入：

```
current_price = 当前价 P_0（从 instrument_context 或 market_report 提取）
target_price_mid = Step 4 综合目标价中位
                  → **必须**用 Step 4 中 compute_weighted_target_price 或
                    compute_overlap_target_price 工具输出的 mid 值
                  → 禁止用"自己重算的调整后中位"或"Step 5 Base case 反推"
threshold_dn_pct = 第一步算出的下沿阈值（如 27 表示 27%）
threshold_up_pct = 第一步算出的上沿阈值（如 63 表示 63%）
target_price_source = "compute_weighted_target_price #N 工具输出"（审计字段）
```

工具返回：
- `deviation_pct`：偏离度（Python 计算，可信）
- `rating`：BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL（机械映射结果）
- `explanation`：完整计算链路

**你的工作**：直接采用工具返回的 `rating`。

#### ⛔ 已经堵的漏洞（禁止重犯）

1. **绕道 #1**：LLM 自己在 Step 6 第二步重新调整 PEG 参数（或其他估值方法的输入参数），算出新的"调整后目标价"，反推新的偏离度。**禁止**——若你认为 Step 4 估值方法选错或参数不合理，必须**回头改 Step 4**（重新调用 compute_pe_eps_target_price 等工具），不允许在 Step 6 内部偷改。

2. **绕道 #2**：用"区间上限/下限"+ 主观百分比判评级。**禁止**——评级只能由 `compute_step6_rating_mapping` 工具输出。

#### 例外情形（可微调评级但必须留痕）

仅以下三种情形允许"按工具得出 X，但主观调整为 Y"：
1. 业绩拐点已经确认衰退/顶部 + 数据完整度 L0/L1 + 红旗 ≥3 → 可下调 1 档
2. 业绩拐点已经确认底部反转 + 数据完整度 L0/L1 + 红旗 ≤1 + Bull anchor 强 → 可上调 1 档
3. consensus.crowded = yes 时按第四步对照表调整

**其他任何主观调整（"短期透支" / "估值消化" / "市场情绪" 等）一律禁止**。

#### 输出格式（强制）

> 工具调用：`compute_step6_rating_mapping`
> 输入：current_price=__, target_price_mid=__（来源：__）, threshold_dn=__%, threshold_up=__%
> 工具返回：deviation_pct=__, rating=**__**
> 是否有例外情形：是/否（如有，说明哪条 + 调整为 Y + 留痕理由）

> **最终评级 R（拥挤度调整 + 对称升降档前）**：BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL

### 第四步：拥挤度调整（在评级输出后强制检查）

⚠️ **核心理念**：拥挤度本身就是**反向风险信号**。拥挤多头 = 大家追多到极致，反向风险是下跌。但**温和降档原则**——只惩罚极端追高（BUY 档），不阻止中度偏多（OVERWEIGHT 在反向风险中仍可接受）。

#### 强制对照表（温和降档版）

**情形 A：consensus.crowded = yes 且 direction = 偏多（拥挤多头）**

| 按估值得出的 R | 是否允许 | 理由 |
|---------------|---------|------|
| BUY | ❌ 禁止 → 降至 OVERWEIGHT | 不能在拥挤多头继续极端追高 |
| OVERWEIGHT | ✅ **保留**（温和降档原则）| 中度偏多在反向风险中仍可接受 |
| HOLD | ✅ 保留 | — |
| UNDERWEIGHT | ✅ **保留**（拥挤多头本身就是反向信号）| 不应该被拉回 HOLD |
| SELL | ✅ **保留** | 同上 |

**情形 B：consensus.crowded = yes 且 direction = 偏空（拥挤空头）**

| 按估值得出的 R | 是否允许 | 理由 |
|---------------|---------|------|
| SELL | ❌ 禁止 → 升至 UNDERWEIGHT | 不能在拥挤空头继续极端追空 |
| UNDERWEIGHT | ✅ **保留**（温和升档原则）| 中度偏空在反向机会中仍可接受 |
| HOLD | ✅ 保留 | — |
| OVERWEIGHT | ✅ **保留**（拥挤空头本身就是反向机会）| 不应该被拉回 HOLD |
| BUY | ✅ **保留** | 同上 |

**情形 C：consensus.crowded = no** —— 无任何调整，按估值得出的 R 直接采纳。

#### 强制示例（温和降档版）

✅ 正确：按估值给 OVERWEIGHT，consensus = 拥挤多头 → **保留 OVERWEIGHT**（温和降档原则）
✅ 正确：按估值给 BUY，consensus = 拥挤多头 → **降至 OVERWEIGHT**（只惩罚极端追高）
✅ 正确：按估值给 UNDERWEIGHT，consensus = 拥挤多头 → **保留 UNDERWEIGHT**
✅ 正确：按估值给 UNDERWEIGHT，consensus = 拥挤空头 → **保留 UNDERWEIGHT**（温和升档原则）
✅ 正确：按估值给 SELL，consensus = 拥挤空头 → **升至 UNDERWEIGHT**（只惩罚极端追空）
❌ 错误：按估值给 OVERWEIGHT，consensus = 拥挤多头 → 降至 HOLD（**旧逻辑双档降，已废弃**）
❌ 错误：按估值给 UNDERWEIGHT，consensus = 拥挤多头 → 拉回 HOLD（反方向理解了规则）

#### 输出格式

- 如果 R 因拥挤度调整改变 → 写明"按估值得出 X，因 consensus = 拥挤[多/空]头，按对照表调整为 Y"
- 如果 R 不变 → 写明"consensus = 拥挤[多/空]头，但按估值得出 X 落在保留区，无调整"

### 第五步：对称升降档修正（强制检查）

之前 RM 只有"降档机制"（拥挤度/数据缺失/红旗都向下），导致"低估值 + 优质 + 拐点确认"票被锁死在 HOLD。本步加**对称升档机制**：

#### 升档条件（所有条件同时满足）

| 条件 | 说明 |
|------|------|
| Step 3 业绩拐点 | "加速期" 或 "底部反转" |
| 数据完整度 | L0 或 L1（VALUATION_METHOD.data_completeness）|
| 红旗数量 | fundamentals.SUMMARY.red_flags ≤ 1 条（且无重大红旗）|
| 估值偏离 | 当前价处于综合区间中位以下（深度低估区间）|
| 反向无拥挤限制 | 不能与拥挤度调整冲突（如拥挤多头时不能升档 OVERWEIGHT → BUY）|
| stock_profile 推荐 | DECISION_STYLE 不是 momentum（动量股不靠估值低估升档）|

→ 评级可上调 1 档（HOLD → OVERWEIGHT，OVERWEIGHT → BUY，UNDERWEIGHT 升 HOLD 不允许 — 升档只对偏多档）

#### 降档条件（多个独立触发，可叠加但最多降 2 档）

| 条件 | 影响 |
|------|------|
| 数据完整度 L3 | -1 档 |
| 红旗 ≥3 条 | -1 档 |
| 拐点 = "顶部" 或 "衰退" | -1 档 |
| Bear 论据 anchor 强 + 业绩可持续性 = "待验证" | -1 档 |

#### 输出格式

```
- 升档检查：[通过/不通过]，[如通过：升至 X / 如不通过：列出不满足的条件]
- 降档检查：[触发数 N，降 N 档至 X / 未触发任何降档]
- 最终 R 修正后：__
```

⚠️ **强制约束**：升档机制不能与拥挤多头规则冲突（即拥挤多头时不能从 OVERWEIGHT 升至 BUY）。升档前必须先过拥挤度对照表。

### 第六步：趋势叠加（4 个子步骤，全部强制工具调用）

⚠️ **核心理念**：以上五步全部按"估值主导"逻辑推导评级——这对 blue_chip 蓝筹股是合理的，但对 high_beta_growth 成长股、theme_speculation 题材股容易"过早 SELL"，错过纯趋势/情绪/催化机会。

参考真实头部投研团队（桥水/Citadel/Tiger 多因子框架）——他们对不同股性叠加不同的"趋势信号通道"：
- **Style 维度**：成长股看动量，题材股看情绪
- **报告权重维度**：股性决定 market/news/sentiment 谁权重更高
- **硬催化维度**：卖方目标价变化、机构持仓变化、北向资金、KOL 趋势

本步**用 3 个独立的趋势信号通道各自给出 ±1 档建议，最后由合成工具 cap 到 ±1 档**。每一步都是工具调用，**完全规则化**。

#### 6.1 Style-Conditional 趋势调整

**必须**调用 `compute_step6_style_adjustment` 工具：

```
rating_after_mechanical = 第五步对称升降档后的评级
style = stock_profile.style
composite_score = QUANT_SCORE.composite
momentum_score = QUANT_SCORE.factor_scores.momentum
```

工具按 style 差异化阈值（已降阈版，对动量更敏感）：
- blue_chip：永不调整
- cyclical：c≥75 + m≥75 → +1 / c≤25 + m≤30 → -1
- high_beta_growth：c≥60 + m≥70 → +1 / c≤40 + m≤35 → -1
- theme_speculation：c≥50 + m≥65 → +1 / c≤50 + m≤45 → -1
- illiquid：c≥65 + m≥65 → +1 / c≤35 + m≤35 → -1
- etf：m≥65 → +1 / m≤35 → -1

工具返回的 `adjustment` 字段（-1/0/+1）即为 **style_adjustment**，传入 6.4 合成。

#### 6.2 非估值方向票调整

把 stock_profile.REPORT_WEIGHTS 真正接入评级——你读完 market/news/sentiment 三份报告后，**为每份报告给出一个方向票（-1 全看空 ~ +1 全看多）**，工具按权重加权后判断是否调整。

**必须**调用 `compute_step6_report_weighted_vote_adjustment` 工具：

```
rating_after_style_adj = 6.1 工具返回的 new_rating
market_weight = stock_profile.REPORT_WEIGHTS.market
news_weight = stock_profile.REPORT_WEIGHTS.news
sentiment_weight = stock_profile.REPORT_WEIGHTS.sentiment
market_direction_vote = 你读 market 报告后的方向票（-1 ~ +1）
news_direction_vote = 你读 news 报告后的方向票
sentiment_direction_vote = 你读 sentiment 报告后的方向票
```

⚠️ **方向票打分规范**：
- +1.0：报告内容**全面看多**（如趋势强、热度高、催化密集）
- +0.5：**偏多但有保留**
- 0：**中性 / 多空平衡**
- -0.5：**偏空但有保留**
- -1.0：报告内容**全面看空**（如趋势破位、热度退潮、利空集中）

工具内部计算加权后 |weighted_vote| ≥ 0.3 → 触发 ±1 档；返回的 `adjustment` 字段为 **vote_adjustment**。

#### 6.3 催化动量调整（硬数据）

从 news / sentiment / fundamentals 报告中**提取 4 个硬数据信号**，工具按规则化打分到 0-100，作为催化动量评分：

**必须**调用 `compute_step6_catalyst_momentum_adjustment` 工具（任一参数缺失填 None）：

```
rating_after_vote_adj = 6.2 工具返回的 new_rating
sell_side_target_change_pct = 近 30 日卖方目标价中位变化百分比（如高盛 5/13 上调 363 → 计算 vs 30 日前中位的变化%）
institutional_holding_change_pct = 近 1 季机构持仓变化百分比（如 Q1 增仓+5%）
northbound_flow_5d_direction = 北向资金近 5 日方向（+1 净流入 / 0 中性 / -1 净流出）
kol_bullish_ratio_trend_pct = KOL 多头率相对 30 日均的变化（如从 60% 升至 75% → +15pp）
```

**数据提取规则**：
- 找不到具体数字时填 `None`，**禁止**编造或瞎估
- 工具要求**至少 2 个**有效参数，否则返回 `skipped`
- 工具内部按阈值打分 → composite 0-100 → ≥70 触发 +1 / ≤30 触发 -1

返回的 `adjustment` 字段为 **catalyst_adjustment**。

#### 6.4 三类信号最终合成

**必须**调用 `compute_step6_adjustment_synthesis` 工具：

```
rating_after_symmetric = 第五步对称升降档后的评级（注意不是 6.3 返回的链式结果）
style_adjustment = 6.1 工具返回的 adjustment（-1/0/+1）
vote_adjustment = 6.2 工具返回的 adjustment（-1/0/+1）
catalyst_adjustment = 6.3 工具返回的 adjustment（-1/0/+1）
```

工具内部合成：取符号（sign(sum)），最终调整 capped 至 ±1 档，应用 no-cross-HOLD。

#### 输出格式（强制）

```
6.1 Style 调整：调用 compute_step6_style_adjustment
    输入：rating=__, style=__, c=__, m=__
    返回：adjustment=__, new_rating=__, rule=__

6.2 非估值方向票：调用 compute_step6_report_weighted_vote_adjustment
    输入：m_weight=__, n_weight=__, s_weight=__, m_vote=__, n_vote=__, s_vote=__
    返回：weighted_vote=__, adjustment=__, new_rating=__

6.3 催化动量：调用 compute_step6_catalyst_momentum_adjustment
    输入：sell_side=__, inst=__, north=__, kol=__
    返回：composite=__, adjustment=__, new_rating=__

6.4 合成：调用 compute_step6_adjustment_synthesis
    输入：style_adj=__, vote_adj=__, catalyst_adj=__
    返回：raw_sum=__, final_adjustment=__, new_rating=__

最终 R（第六步后）：__
```

### 第七步：极端背离防御（强制兜底）

⚠️ **作为最后一道保护**——三类趋势信号合成后，仍可能出现"评级方向与量化锚极端背离"的边缘情形。本步是兜底，只在**极端不一致**时触发。

#### 极端背离判定表

| 第六步后评级 | 触发条件 | 强制调整 |
|------------|---------|---------|
| **BUY / OVERWEIGHT** | composite ≤ 20 | 强制降至 HOLD（量化层面极度警示）|
| **UNDERWEIGHT / SELL** | composite ≥ 80 | 强制升至 HOLD（量化层面极度乐观）|
| 其他 | 不触发 | 保留第六步评级 |

**非极端背离不再触发任何调整**——中度差异已经在第六步处理过，避免重复加权 quant 信号。

#### 例外

- 业绩拐点确认（刚出超预期 Q1 业绩）→ 说明量化锚滞后于新数据，不触发
- 地雷排查命中（第负一步）→ 直接 SELL，跳过本步

#### 输出格式

```
- 第六步后评级：__
- QUANT_SCORE.composite：__/100
- 是否触发极端背离：是/否
- 处理：[保留 / 强制调整为 __]
- 最终 R（第七步后）：__
```

### 评级 COT 示例（参考格式，禁止照抄结论）

> 综合 Step 1-5 分析，当前价 248 元处于综合目标价区间 [200, 230] 元的上沿之外（高 8%）；业绩拐点判断为"加速期但 Q2 是关键验证点"，可持续性"待验证"；Base case 目标价 200 元（55% 概率）意味着持有当前价格存在 -19% 的预期回报；3 种估值方法（PEG/PE×EPS/同业可比）都指向 180-230 元区间，方向一致。
> 给出 UNDERWEIGHT 评级。理由：(1) 估值脱离 3 种方法的合理共识区间 (2) Base case 隐含负向预期 (3) 业绩拐点 Q2 才能验证，当前价格已透支验证后的乐观预期。
> 为什么不给 SELL：(1) 业绩拐点尚未被证伪 (2) Bull case 有 20% 概率上行 21% (3) 公司护城河（行业龙头）支持中长期估值溢价。
> 为什么不给 HOLD：(1) Base case 隐含 -19% 不能视为合理估值 (2) 多元估值交叉一致指向偏离。

---

## Step 7: 评级置信度

| 等级 | 触发条件 |
|------|---------|
| **高** | 3 种估值方法目标价相对偏离 < 15% + 业绩拐点明确（数据指向同一方向） + 数据完整度 L0/L1 |
| **中** | 3 种估值方法相对偏离 15-30% / 拐点判断有不确定性 / 数据完整度 L2 |
| **低** | 估值方法分歧大（> 30%）/ 拐点完全不明 / 数据完整度 L3 / 多空双方都有强证据但方向不明 |

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

## 关键约束（最后强调）

- **评级是 Step 6 的综合判断产出**，不准退化为"d 阈值定评级"
- **多元估值至少 3 种**，少于 3 种必须明确说明"不适用"原因
- **Base case 是真实预期**，不是中位数
- **拥挤度调整必须在评级输出后显式标注**
- **COT 必须透明**：每步都有可追溯的结构化输出，PM 能复盘你的思考路径

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
  deviation_pct: <float>                 # Step 6 第二步偏离度
  threshold_dn_pct: <float>              # Step 6 第一步动态阈值下沿
  threshold_up_pct: <float>              # Step 6 第一步动态阈值上沿
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
