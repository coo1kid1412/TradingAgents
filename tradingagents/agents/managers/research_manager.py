import logging
import json

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from tradingagents.agents.utils.agent_utils import build_instrument_context, RISK_DEBATE_PHRASING_RULES
from tradingagents.agents.managers.rm_tools import RM_TOOLS, RM_TOOLS_BY_NAME

logger = logging.getLogger(__name__)

# 工具调用循环上限——避免 LLM 反复调同一工具陷入死循环
_MAX_TOOL_ITERATIONS = 10


def _run_tool_calling_loop(llm_with_tools, initial_messages):
    """执行 LLM 工具调用循环，直到 LLM 不再调工具或达到上限。

    返回最终 AIMessage（含完整 thesis 文本）。
    """
    messages = list(initial_messages)
    for iteration in range(_MAX_TOOL_ITERATIONS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            # LLM 不再调工具，循环结束
            logger.info("RM tool calling 循环结束（第 %d 轮，无 tool_calls）", iteration + 1)
            return response

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
    return final


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"], state.get("company_name", ""))
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        consensus_snapshot = state.get("consensus_snapshot", "")
        stock_profile = state.get("stock_profile", "")

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{fundamentals_report}\n\n{market_research_report}\n\n{news_report}\n\n{sentiment_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=3)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

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

你已经绑定了 9 个计算工具。**所有数值计算必须通过工具调用完成，禁止在报告里"心算"或"凭感觉"算**。

| 计算场景 | 必须调用的工具 |
|---------|-------------|
| Step 4 估值方法 1 PE×EPS | `compute_pe_eps_target_price` |
| Step 4 估值方法 PEG | `compute_peg_target_price` |
| Step 4 综合目标价区间（严格重叠）| `compute_overlap_target_price` |
| Step 4 综合目标价（加权折中，当无严格重叠时）| `compute_weighted_target_price` |
| Step 5 三情景概率加权 E | `compute_scenario_weighted_e` |
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

### 输出要求（每种方法独立）

> **方法 1: [方法名]**
> - 输入数据：[列出所有用到的数据 + 来源]
> - 计算过程：[公式展开]
> - 目标价：__ 元
> - 该方法的局限性：[1 句话]
>
> **方法 2: [方法名]**
> - ...
>
> **方法 3: [方法名]**
> - ...

### 综合目标价

> | 方法 | 目标价 | 权重 |
> |------|--------|------|
> | 方法 1 | __ | __ |
> | 方法 2 | __ | __ |
> | 方法 3 | __ | __ |
>
> **综合目标价区间**：[低位] 至 [高位] 元
> （区间 = 3 种方法的合理重叠区域；不重叠则取相对均匀的覆盖区间，并说明分歧原因）

### 重要约束

- **PEG 半衰公式**：预期 EPS = EPS_TTM × (1 + min(增长率, 60%)/2)——防止异常高增长被纯线性外推
- **数据缺失**：如某方法关键输入缺失（如 fundamentals.SUMMARY 的 EPS=null），明确写"该方法不适用，原因：__"，不强行外推
- **禁止凭感觉给目标 PE/PB**：必须引用 stock_profile.target_pe_range / consensus.market_implied_pe_range / 行业可比数据，其中至少一个
- **置信度对照**：综合目标价区间宽度 > 当前价 × 50% → 标"低置信度"

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
- 三情景目标价应大致覆盖 Step 4 综合目标价区间
- 核心假设必须可观测可证伪（"业绩超预期"不可接受，"Q2 营收 >25%"可接受）

**Base case 是真实预期，不是中位数**——你认为最可能发生的路径，不是 Bull 和 Bear 的平均。

---

## Step 6: 综合评级判断（COT 主观）

**评级不是公式产出，是 RM 基于 Step 1-5 综合判断后的主观决定**。

写一段 200-300 字的"评级 COT"，必须包含以下要素：

1. **当前价 vs 综合目标价区间**的位置（在区间下沿/区间内/区间上沿/超出区间）
2. **业绩拐点判断**对评级的支撑/反对
3. **Base case 目标价**是否对当前价格有吸引力
4. **多元估值交叉一致性**（3 种方法是否一致指向同一方向）
5. **如果偏离行业典型估值范式**，说明为什么这么做
6. **反方推理**：明确回答"为什么不给更激进/更保守的评级"

### 评级（5 档之一）

> **最终评级 R**：BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL

### 拥挤度调整（在评级输出后强制检查）

⚠️ **核心理念**：拥挤度本身就是**反向风险信号**。拥挤多头 = 大家追多到极致，反向风险是下跌；拥挤空头 = 大家空到极致，反向风险是反弹。因此：
- **拥挤多头时，做空（UNDERWEIGHT/SELL）是合理的反向操作**，不应被规则拦下
- **拥挤空头时，做多（OVERWEIGHT/BUY）是合理的反向机会**，不应被规则拦下

#### 强制对照表（按 consensus.crowded 和 direction 查表，禁止其他理解）

**情形 A：consensus.crowded = yes 且 direction = 偏多（拥挤多头）**

| 按估值得出的 R | 是否允许 | 理由 |
|---------------|---------|------|
| BUY | ❌ 禁止 → 降至 OVERWEIGHT | 不能在拥挤多头继续追高 |
| OVERWEIGHT | ❌ 禁止 → 降至 HOLD | 同上 |
| HOLD | ✅ 保留 | — |
| UNDERWEIGHT | ✅ **保留**（拥挤多头本身就是反向信号，UNDERWEIGHT 是合理反向） | 不应该被拉回 HOLD |
| SELL | ✅ **保留** | 同上 |

**情形 B：consensus.crowded = yes 且 direction = 偏空（拥挤空头）**

| 按估值得出的 R | 是否允许 | 理由 |
|---------------|---------|------|
| SELL | ❌ 禁止 → 升至 UNDERWEIGHT | 不能在拥挤空头继续追空 |
| UNDERWEIGHT | ❌ 禁止 → 升至 HOLD | 同上 |
| HOLD | ✅ 保留 | — |
| OVERWEIGHT | ✅ **保留**（拥挤空头本身就是反向机会，OVERWEIGHT 是合理反向） | 不应该被拉回 HOLD |
| BUY | ✅ **保留** | 同上 |

**情形 C：consensus.crowded = no** —— 无任何调整，按估值得出的 R 直接采纳。

#### 强制示例（必须严格遵守，不准重蹈覆辙）

✅ 正确：按估值给 UNDERWEIGHT，consensus = 拥挤多头 → **保留 UNDERWEIGHT**（拥挤多头不阻止偏空评级）
✅ 正确：按估值给 OVERWEIGHT，consensus = 拥挤多头 → **降至 HOLD**（拥挤多头禁止追高）
✅ 正确：按估值给 OVERWEIGHT，consensus = 拥挤空头 → **保留 OVERWEIGHT**（拥挤空头不阻止偏多评级，反而支持反向机会）
✅ 正确：按估值给 SELL，consensus = 拥挤空头 → **升至 UNDERWEIGHT**（拥挤空头禁止追空）
❌ 错误：按估值给 UNDERWEIGHT，consensus = 拥挤多头 → 拉回 HOLD（**反方向理解了规则**——拥挤多头不是"禁止偏空"，是"禁止偏多"）

⛔ **绝对禁止**：在拥挤多头时把 UNDERWEIGHT/SELL 拉回 HOLD。这是把"反向风险信号"误读成了"反向操作禁令"，逻辑完全反了。

#### 输出格式

- 如果 R 因拥挤度调整改变 → 写明"按估值得出 X，因 consensus = 拥挤[多/空]头，按对照表调整为 Y"
- 如果 R 不变 → 写明"consensus = 拥挤[多/空]头，但按估值得出 X 落在保留区，无调整"

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

## 历史教训
\"{past_memory_str}\"

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
