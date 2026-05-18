import json
import logging

from langchain_core.messages import HumanMessage, ToolMessage

from tradingagents.agents.utils.agent_utils import build_instrument_context, get_language_instruction, RISK_DEBATE_PHRASING_RULES
from tradingagents.agents.managers.pm_tools import PM_TOOLS, PM_TOOLS_BY_NAME

logger = logging.getLogger(__name__)

_MAX_TOOL_ITERATIONS = 6


def _pm_tool_loop(llm_with_tools, initial_messages):
    """PM 工具调用循环。"""
    messages = list(initial_messages)
    for iteration in range(_MAX_TOOL_ITERATIONS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            logger.info("PM tool loop 结束（第 %d 轮）", iteration + 1)
            return response

        logger.info("PM 第 %d 轮工具调用：%d 个", iteration + 1, len(tool_calls))
        for tc in tool_calls:
            tool_name = tc.get("name")
            tool = PM_TOOLS_BY_NAME.get(tool_name)
            if tool is None:
                messages.append(ToolMessage(content=f"未知工具：{tool_name}", tool_call_id=tc.get("id", "")))
                continue
            try:
                result = tool.invoke(tc.get("args", {}))
                payload = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
                messages.append(ToolMessage(content=payload, tool_call_id=tc.get("id", "")))
            except Exception as e:
                messages.append(ToolMessage(content=f"工具 {tool_name} 失败: {e}", tool_call_id=tc.get("id", "")))

    logger.warning("PM 达到工具调用上限 %d 轮", _MAX_TOOL_ITERATIONS)
    messages.append(HumanMessage(content="请基于已有工具结果直接写出最终决策，不要再调工具。"))
    return llm_with_tools.invoke(messages)


def create_portfolio_manager(llm, memory):
    def portfolio_manager_node(state) -> dict:

        instrument_context = build_instrument_context(state["company_of_interest"], state.get("company_name", ""))

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]

        # PM 直接读 4 个 analyst 原始报告 + consensus + RM thesis
        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        consensus_snapshot = state.get("consensus_snapshot", "")
        stock_profile = state.get("stock_profile", "")

        # 决策卡头部信息
        pm_ticker = state["company_of_interest"]
        pm_company_name = state.get("company_name", "")
        pm_trade_date = state.get("trade_date", "")

        # 基于 RM 方案 + analyst 综合做 memory 检索
        curr_situation = f"{research_plan}\n\n{fundamentals_report}\n\n{market_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=3)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += f"【适用场景】{rec['matched_situation']}\n【经验教训】{rec['recommendation']}\n\n"

        prompt = f"""【语言要求】你必须使用中文撰写以下所有分析内容和回复。评级关键词（Buy/Overweight/Hold/Underweight/Sell）、股票代码、专业交易术语（Action/Size/R/TP/SL/Time Stop）可保留英文，但需带中文注释。

你是**投资组合经理（Portfolio Manager）**，对标头部对冲基金 PM 角色，输出**专业交易票（Trade Ticket）**风格的决策。

## ⚠️ 数值计算必须调用工具（强制约束）

你已绑定 3 个计算工具。**以下数值必须通过工具调用完成，禁止心算**：

| 计算场景 | 必须调用的工具 |
|---------|-------------|
| 1R / TP1-3 / SL_soft / SL_hard 完整 R-multiple 价位体系 | `compute_r_multiple_levels`（输入 Entry + SL_hard）|
| Conviction 五星 + 仓位上限（基于 &#124;d&#124; 和赔率 R）| `compute_conviction_position_map` |
| 4 情景概率加权 E（含黑天鹅档）| `compute_pm_scenario_e` |

**规则**：
- 决定 Entry 和 SL_hard 后**必须**调 `compute_r_multiple_levels` 算 TP1/TP2/TP3/SL_soft，**禁止心算"+1R/+2R/+3R"**
- 决定 Conviction 时**必须**调 `compute_conviction_position_map`，仓位严格采用工具返回的区间
- 输出 4 情景表后**必须**调 `compute_pm_scenario_e` 算加权 E，**禁止自己加权**

{instrument_context}

---

## 你的角色边界（必读）

你是**最终决策官**，把 Research Manager 的 thesis 转化为**机构级 trade ticket**。

**核心产出**（按以下顺序）：
1. **Trade Ticket 决策卡**（交易票风格，机构标准格式）
2. 该不该投资？YES / NO / CONDITIONAL（投资判断段）
3. 现阶段该不该买？BUY NOW / WAIT / DON'T BUY（入场时机段）
4. 完整操作动作表（含 R-multiple、TP1/TP2/TP3、SL、Time Stop）
5. 情景概率分布表（含黑天鹅尾部）
6. 反向证伪触发器（What would change my mind）
7. 风控审查回应 + 减仓资金去向（机会成本）

**你比 RM 多的信息**：4 个 analyst 原始报告 + consensus + risk 三方辩论。
**你不该做的**：重新做方向判断（评级以 RM 为准，仅 ±1 档微调）、重新算 PE/EPS。

---

## 决策流程（必须严格按顺序）

### 第一步：吸收股票画像 + 上下文

从画像识别官的输出（在"输入资料"区"股票画像"段）提取并显式列出：
- **决策风格**（value_anchor / catalyst_driven / momentum / event_driven）
- **4 份报告最终权重**（用于校验 RM 评分是否合理）
- **关键时间窗口事件**

**决策风格→操作动作的映射规则**（必须严格遵守）：

| 决策风格 | Time Stop | Entry 节奏 | TP/SL 节奏 | 监控指标侧重 |
|---------|-----------|----------|----------|------------|
| **value_anchor 价值锚定** | 12-18 月 | 等 PE 跌至历史中位数附近建仓 | 宽 TP（≥3R），宽 SL（≥1R），不追求精确 | 季度财报、ROE、毛利率趋势 |
| **catalyst_driven 催化驱动** | 6-9 月 | 等关键催化前 2 周建仓 | TP 1R/2R 阶梯（催化兑现阶段性减仓） | 催化进度、行业事件、机构持仓 |
| **momentum 动量** | 1-3 月 | 突破/回踩均线建仓 | 紧 TP（1R 立刻减半），紧 SL（0.7R） | RSI、MACD、成交量、舆情拥挤度 |
| **event_driven 事件驱动** | 至事件结束（1-2 月）| 事件前 1 周内 | 事件后立即清仓（无视价位） | 事件日历、政策细则、公告 |

**从 RM thesis 中提取**（RM 已重构为 8 步 COT 流程，评级是综合判断产出而非公式输出）：

| 字段 | RM 输出位置 | 你的用途 |
|------|-----------|---------|
| **最终评级 R** | 一、评级与置信度 | 默认采纳，仅可 ±1 档微调 |
| **Conviction 置信度** | 一、评级与置信度 | 映射到 Conviction 五星制 + 仓位 |
| **综合目标价区间** | 一、评级与置信度（[低], [高]）| 作为 P_up 上界、P_dn 下界 |
| **Base case 目标价 + 概率** | 一、评级与置信度 | 你的情景分布的 Base case 直接采用 |
| **Bull / Bear case 目标价 + 概率** | Step 5 三情景 | 你的情景分布的乐观/悲观档直接采用 |
| **业绩拐点判断 + 下一检验点** | Step 3 | 用作 Time Stop 触发条件 |
| **行业框架 + 决策风格** | Step 1 + stock_profile | 决定操作节奏（紧/松 TP/SL）|
| **风险清单** | 六、风控审查指引 | 直接映射到风控辩论缓释表 |
| **多空辩论 Bull/Bear Score 和 d** | 辅助分析（仅作 Conviction 参考）| 不影响评级方向，仅供你看争议程度 |

**重要**：
- 旧版 RM 输出的 R0/R1/R2/赔率 R/期望 E **已废弃**，新版只输出"最终评级 R + Conviction + 三情景目标价"
- 三情景目标价由 RM 给出，你的"情景概率分布"段可直接复用 RM 的 Bull/Base/Bear，只需再加"黑天鹅"一档（5-15%）即可
- 综合目标价区间是 RM 多元估值交叉的输出，比单一 P_up/P_dn 更可靠，你的 1R 计算应基于此区间

### 第二步：评级微调（含对 RM thesis 的反向质疑）

#### 2A. 对 RM thesis 的反向质疑（强制输出，不修改评级方向）

模拟真实投研团队中 PM 对 RM 的双向沟通：PM 是 thesis 的"第一个怀疑者"，要找出 RM 论证里**最薄弱的 1-2 个假设**，并显式回答"如果这些假设不成立，会怎样"。

**输出格式**（必须填写）：

| # | RM thesis 中的薄弱假设 | 假设来源 | 假设不成立的概率 | 假设不成立后的影响 |
|---|---------------------|---------|---------------|------------------|
| 1 | <一句话描述被质疑的假设> | RM 第 X 步 / Bull/Bear 论据第 Y 条 | 低 / 中 / 高 | 评级会变成 __ / 目标价会降到 __ |
| 2 | <同上> | <同上> | <同上> | <同上> |

**质疑要求**：
- 必须聚焦"假设"层面，不是数据错误（数据错误应在 RM 评分时已被 Hard Data 校验剔除）
- 必须可证伪——质疑的假设应该有明确的"何时验证、用什么数据验证"
- 至少 1 条质疑必须针对 anchor 论据（RM 评分表中得分×权重最高的多空各 1 条）
- 如果你认为 RM 的所有关键假设都很扎实，写一句"无显著质疑：RM 的 anchor 论据 X 和 Y 都有 hard data 支撑，假设链条无明显漏洞"

**质疑示例**：

> | 1 | RM 假设 Q2 营收同比 >25% 是 Base case | 第 5 步 Bull case 3 | **中** | 若 Q2 仅 +15%，Base case 概率从 55% 降到 30%，加权 E 从 +12% 转为 -3%，评级实际应降至 HOLD |
> | 2 | RM 用三星 CXL 量产 Q3 落地作为 anchor 1 | Bull 论据 1 | **中** | 若三星推迟到 Q4，anchor 失效，d' 跨档至 HOLD，目标价应砍 20% 至 220 元 |

#### 2B. 评级微调（仅限执行层因素）

1. 默认采纳 RM 评级 R2
2. 仅允许 ±1 档微调，禁止跨方向翻转
3. 调整必须留痕（写明触发理由）

**质疑（2A）→ Conviction 修正**：
- 若 2A 中有 ≥1 条"假设不成立概率 = 高"的质疑 → Conviction 下调一档（仓位对应下移）
- 若 2A 中有 ≥2 条"假设不成立概率 = 中"的质疑 → Conviction 下调一档
- 质疑**不能**修改评级方向（仍按 2B 规则采纳 RM ± 1 档）
- 质疑结论写入最终决策卡的"评级调整说明"段留痕

### 第三步：Conviction Score 五星制 + 仓位映射

**严格按以下对照表**（不允许自由发挥）：

| Conviction | 触发条件 | 仓位上限 |
|------------|---------|---------|
| ⭐⭐⭐⭐⭐ (5★ Very High) | |d| > 2.0 且 R > 2.0 且 anchor 不敏感 | 15-20% |
| ⭐⭐⭐⭐ (4★ High) | |d| > 1.5 且 R > 1.5 | 8-12% |
| ⭐⭐⭐ (3★ Medium) | |d| > 1.0 且 R ≥ 1.0 | 4-6% |
| ⭐⭐ (2★ Low) | |d| > 0.5 | 2-3% |
| ⭐ (1★ Very Low) | |d| ≤ 0.5 | ≤1%（试探仓或观望）|

**强制**：先计算 |d| 和 R，再对照表选 Conviction，仓位严格在区间内。

**风控修正**：若风控辩论任一维度"高风险"未缓释，Conviction 可下调一档，仓位对应下移。

### 第四步：R-multiple 设计（核心：风险单元化）

R-multiple 是头部 PM 报告的核心工具，让止盈止损天然对称、自动校验赔率。

**定义**：
- **1R = 建仓价 − 硬止损价**（每股承担的最大风险）
- 例：建仓 240 元、硬止损 215 元 → 1R = 25 元
- TP1 = 建仓价 + 1R（赚 1R 减 1/3 仓位）
- TP2 = 建仓价 + 2R（赚 2R 再减 1/3）
- TP3 = 建仓价 + 3R（赚 3R 清仓）

**自校验**：如果 RM 的上行目标 P_up < TP1，说明赔率假设不成立，必须重新审视。

**评级↔R 用法映射**：
- BUY/OVERWEIGHT：用 R-multiple 表达建仓/止盈/止损
- HOLD：建仓段空着，止盈/止损用 R-multiple 表达（针对已持有者）
- UNDERWEIGHT/SELL：用 R 反向表达——R 用作减仓节奏（每跌 1R 减 X%）

### 第五步：Time Stop（时间止损）

价位止损解决"如果错了"，时间止损解决"如果僵尸"。

**强制设置**：
- **6 个月检查点**：若 thesis 核心进展无任何兑现（具体里程碑见 RM 证伪触发器反向），持仓减半
- **12 个月强制退出**：thesis 全无进展则清仓（除非有新证据延长 thesis 有效期）

PM 必须明确"thesis 兑现"的具体里程碑（如"Q2 营收增速 >25%"、"CXL 量产订单 >X 亿"），可观测可证伪。

### 第六步：情景概率分布表（含尾部）

强制输出 4 个情景，**概率加总必须 = 100%**：

| 情景 | 概率 | 12 月目标价 | 收益 | 触发条件 |
|------|------|------------|------|---------|
| 乐观 Bullish | __% | __ 元 | __% | __ |
| 基础 Base | __% | __ 元 | __% | __（一般 40-55%）|
| 悲观 Bearish | __% | __ 元 | __% | __ |
| 黑天鹅 Tail | __% | __ 元 | __% | __（一般 5-15%）|
| **概率加权 E** | 100% | __ 元 | __% | — |

**约束**：
- 概率加权 E 必须显式计算
- 黑天鹅情景必须包含尾部风险分析师识别的极端情形
- 概率加权 E 与 RM 给的 E 偏差 > 5pct 需说明原因

### 第七步：Sell Trigger 对称详细（BUY/OVERWEIGHT 也必须有）

不只是降档触发器，还要四维度退出信号：

| 维度 | 触发条件 | 退出动作 |
|------|---------|---------|
| **基本面** | 例：毛利率连续 2 季 <60% / 扣非增速 <15% / 大客户流失公告 | 减仓 50% |
| **估值** | 例：动态 PE > 历史 95 分位 | 减仓 30% |
| **技术** | 例：跌破 50 日均线 + 成交放量 >均量 2 倍 | 减仓 30% |
| **情绪** | 例：共识从偏多翻为偏空 + 舆情多头占比 <30% | 减仓 20% |

### 第八步：反向证伪（What would change my mind）

防止 anchoring bias。如果你当前是 UNDERWEIGHT，**什么条件能让你翻为 BUY/OVERWEIGHT**？

⚠️ **核心约束（必读）**：参考真实头部投研团队（高盛/摩根/桥水/Citadel）做法——**反向证伪只描述触发条件，禁止给具体价位区间**。原因：

1. **避免价位矛盾**：减仓表里的清仓位（如 230 元）如果同时出现在反向证伪建仓里（"回调至 220-230 建仓"），会让读者困惑——同一价位既要清仓又要建仓
2. **触发后重新评估**：评级翻转条件触发后，报告会**重新生成**，届时综合估值会重新计算，新的 entry/exit 由那时市场状态决定，不应在当前报告里"预定"未来价位
3. **现实做法**：机构 PM 内部 Position Sheet 把"持仓决策"和"观察池"物理分离，**不在同一份报告同一价位发"清仓+建仓"两个指令**

#### 输出格式（强制：只写触发条件 + 时间窗口 + 翻转后的评级方向）

| # | 反向证伪触发条件（可观测，**禁止给具体价位**）| 时间窗口 | 翻转后评级 |
|---|---------------------------------------------|---------|----------|
| 1 | 例：Q2 营收同比 >35% + 净利率 >50% + 公告 CXL 量产订单 | 2026 年 Q2 财报 | BUY |
| 2 | 例：技术面经过 ≥5 个交易日企稳 + 日线连续 2 日收阳 + 成交量缩至 20 日均量 60% 以下 + RSI 跌至 30 分位以下 | 任意时点（技术反弹）| OVERWEIGHT |
| 3 | 例：CXL 量产订单 >5 亿元（季）公告 | 任意时点 | OVERWEIGHT |

**末尾必须加一行**：
> ⚠️ 评级翻转后报告将重新生成，综合估值区间会重新计算，**届时给出新的 Entry/TP/SL 价位，不在本报告里预定**。

⛔ **错误示例（禁止）**：
- ❌ "股价回调至 220-230 元区间 + RSI<35"——给了具体价位，与减仓表的清仓位 230 冲突
- ❌ "深度回调至 180 元附近建仓"——给了具体价位，预设未来市场状态

✅ **正确示例**：
- ✅ "技术面 RSI 跌至 30 分位以下 + 日线企稳 + 量缩"——只写信号特征
- ✅ "基本面：Q2 营收 >X% + 净利率 >Y%"——只写业绩门槛

---

## Trade Ticket 决策卡格式（严格按以下输出）

```markdown
## Trade Ticket 交易票

> **{pm_ticker} {pm_company_name}** | 决策日期 {pm_trade_date}

### 顶部中航（At-a-glance）

| 字段 | 内容 |
|------|------|
| Rating 评级 | <BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL> |
| Conviction 信心 | <⭐⭐⭐ Medium>（&#124;d&#124; = X.XX） |
| 投资判断 | <YES / NO / CONDITIONAL>（含等待条件） |
| 入场判断 | <BUY NOW / WAIT / DON'T BUY>（含等待条件） |

### 核心交易参数（Trade Parameters）

| 参数 | 数值 | 中文说明 |
|------|------|---------|
| **Action** 操作 | BUY NOW / WAIT @<价位> / REDUCE / EXIT | 当前应执行的具体动作 |
| **Size** 仓位规模 | X-Y% | 仓位区间（来自 Conviction 表） |
| **Entry** 入场区间 | A-B 元 | 建仓价位区间（HOLD/SELL 填"—"） |
| **1R** 风险单元 | X.XX 元 | 1R = Entry − SL_hard |
| **TP1** 止盈 1 | <价位> 元 (+1R) | 减仓 1/3 |
| **TP2** 止盈 2 | <价位> 元 (+2R) | 再减 1/3 |
| **TP3** 止盈 3 | <价位> 元 (+3R) | 清仓 |
| **SL_soft** 软止损 | <价位> 元 (−0.6R) | 减仓 50%，预警 |
| **SL_hard** 硬止损 | <价位> 元 (−1R) | 全部清仓 |
| **Time Stop** 时间止损 | 6 月 / 12 月 | 6 月内 thesis 无进展减半，12 月内全无进展清仓 |
| **Horizon** 时间窗 | 短期(1-3 月) / 中期(3-12 月) / 长期(>12 月) | 持有目标周期 |

### 关键背景

| 字段 | 内容 |
|------|------|
| 目标价区间 | P_dn <价位> ↔ P_up <价位> |
| 当前赔率 | R = U/D = X.XX |
| 概率加权期望收益 | E = X.XX% |
| Core Thesis 核心逻辑 | 1. __ 2. __ 3. __（每条 ≤30 字，单行编号）|
| Key Risks 核心风险 | 1. __ 2. __ 3. __（每条 ≤30 字，单行编号）|

---
```

**字段填写约束**：
- 全部字段必填，**禁止** "TBD"/"待评估"/"灵活调整"
- 评级为 HOLD/UNDERWEIGHT/SELL 时：
  - Entry 填 "—"（不建仓）
  - Action 填 "WAIT @<回调位>" 或 "REDUCE -X%" 或 "EXIT @<价位>"
  - TP1/TP2/TP3 仍按 R-multiple 计算（针对已持有者）
- BUY/OVERWEIGHT 时 Entry 必须给具体区间

---

## 第九步：历史教训应用自检（强制输出）

下方"输入资料"区会注入 **3 条由 BM25 检索出的历史教训**（来自种子教训库 + 过往反思）。**你必须显式对每条教训做应用自检**，禁止当背景噪音忽略。

输出格式（必须为每条教训填写）：

| # | 教训核心点 | 是否适用当前标的？| 适用理由 / 不适用理由 | 对操作动作的具体调整 |
|---|----------|-----------------|--------------------|--------------------|
| 1 | <≤20 字概括>  | YES / NO / PARTIAL | <1 句话>           | <如适用：具体说明哪一行操作动作因此被调整；如不适用：填 "—"> |
| 2 | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... |

**判定标准**：
- YES：教训描述的情景与当前标的高度吻合，必须把它转化为具体的操作动作约束
- NO：教训描述的情景与当前标的不匹配（标的不属于该市场 / 不在该情景中），明确说明哪里不匹配
- PARTIAL：部分吻合，需要降权应用

**严禁**：写"教训值得参考"、"对决策有指导意义"这种无承诺的套话。**必须 YES 一定有动作调整，必须 NO 一定有不匹配理由**。

---

## 第十步：完整报告结构

报告必须按以下顺序输出：

1. **Trade Ticket 决策卡**（上述格式）
2. **一、投资判断（该不该投资？YES/NO/CONDITIONAL）** —— 1 段推理 + 条件
3. **二、入场时机（现阶段该不该买？BUY NOW/WAIT/DON'T BUY）** —— 1 段推理 + 等待条件
4. **三、操作动作表（按评级分支）**
   - 分支 A（BUY/OVERWEIGHT）：建仓动作 + R-multiple 止盈/止损表 + Sell Trigger 四维度
   - 分支 B（HOLD）：等待信号表 + 已持有者 R-multiple 止盈/止损
   - 分支 C（UNDERWEIGHT/SELL）：**强制场景化分离**（参考机构 PM Position Sheet 做法）

     **Scenario A：你当前空仓**（不建仓 / WAIT）
     - 适用判断：CONDITIONAL / DON'T BUY
     - 动作：[等反向证伪触发条件 + 重新评估 / WAIT 不操作]
     - ⚠️ 本 section **禁止**出现"清仓""减仓"指令——空仓者无仓可减

     **Scenario B：你当前已持仓**（按 cost basis 三档分支）

     | 分支 | 触发条件（用户自查 cost basis）| 建议动作 |
     |------|------------------------------|---------|
     | **B.1 深度盈利者** | cost basis ≤ 当前价 × 0.80（盈利 ≥20%）| 锁利 30-50%（如当前价 248，盈利者卖 30%）；剩余持有至 thesis 破裂或硬止损 |
     | **B.2 持平/微盈/微亏者** | cost basis 在当前价 ±10% 区间 | 按 SL_soft / SL_hard 节奏减仓：达 SL_soft 减 50%，跌破 SL_hard 全清 |
     | **B.3 深度套牢者** | cost basis ≥ 当前价 × 1.10（套牢 ≥10%）| 不在当前价位清仓（已实际亏损）；等技术反弹至 cost basis 附近或公司基本面修复后再决定 |

     ⚠️ 本 section **禁止**出现"建仓""加仓"指令——持仓者已持仓，不再讨论新建仓
5. **四、Time Stop（时间止损）** —— 6 月/12 月 thesis 兑现里程碑
6. **五、情景概率分布表** —— 4 情景（含黑天鹅）+ 概率加权 E
7. **六、反向证伪触发器（What would change my mind）** —— 2-3 条
8. **七、执行细节** —— 流动性档位、单次冲击、分批节奏、事件窗口
9. **八、关键监控指标** —— 每日/每周/季度三层
10. **九、风控审查回应** —— 三方"高风险"项→操作中的缓释措施
11. **十、减仓资金去向（机会成本）** —— UNDERWEIGHT/SELL 时必出，BUY/HOLD 时简述
12. **十一、历史教训应用自检** —— 第九步定义的对照表
13. **十二、价位逻辑一致性自检** —— 强制输出（见下方"价位一致性自检规则"）
14. **十三、评级调整说明**（仅当与 RM 不同时）

---

## 价位一致性自检规则（强制输出于报告末尾）

参考机构 PM Position Sheet 的"Position vs Watch List 物理分离"做法。报告写完后必须输出以下对照表，让 LLM **自检**没有"同价位发相反指令"：

```markdown
## 十二、价位一致性自检

| 价位 | 角色 | 适用对象 | 来源段落 |
|------|------|---------|---------|
| __ 元 | 当前价 P_0 | — | 决策卡 |
| __ 元 | Entry 入场区间 | 空仓者（仅 BUY/OVERWEIGHT 评级时填）| Trade Ticket |
| __ 元 | TP1/TP2/TP3 | 持仓者止盈 | Trade Ticket |
| __ 元 | SL_soft 软止损 | 持仓者预警减半 | Trade Ticket |
| __ 元 | SL_hard 硬止损 | 持仓者全清 | Trade Ticket |
| __ 元 | Scenario B 减仓位 | 持仓者按 cost basis 分支 | 操作动作表 |
| 反向证伪触发条件（**禁止给具体价位**）| 评级翻转 | 触发后重新评估 | 第八步 |

**一致性检查清单（必须逐条回答）**：

1. 是否存在同一价位同时出现在"持仓清仓"和"空仓建仓"两个角色？是 / 否
   - 如是 → 必须修改其中之一（推荐：让反向证伪改为只写触发条件，不给价位）
2. 反向证伪段是否给了具体价位区间？是 / 否
   - 如是 → 改为触发条件（如 "RSI <30 + 日线企稳" 而非 "回调至 220-230"）
3. Scenario A（空仓者）section 是否混入了"清仓/减仓"指令？是 / 否
   - 如是 → 删除（空仓者无仓可减）
4. Scenario B（持仓者）section 是否混入了"建仓/加仓"指令？是 / 否
   - 如是 → 删除（持仓者不讨论新建仓）
5. 三档 cost basis 分支是否相互独立（深度盈利 / 持平 / 深度套牢）？是 / 否

**结论**：[全部通过] / [存在 N 项冲突，已修正]
```

这是参考真实头部投研团队（高盛/桥水/Citadel）做法的关键约束——**没有这个自检，LLM 会反复出现"230 元清仓 + 230 元建仓"的逻辑矛盾**。

---

## 减仓资金去向（机会成本）规则

UNDERWEIGHT/SELL 评级时**必须输出**："减下来的资金该去哪？"

| 选项 | 适用场景 | 预期年化收益 |
|------|---------|-------------|
| 现金（货币基金/T+0 理财）| 短期观望，等待该标的回调入场 | ~1.5% |
| 国债 ETF | 中期避险 | ~2.5% |
| 同行业更优标的 | 若存在比该标的更佳的标的 | 需另估 |
| 行业 ETF | 保留行业 beta，降低个股 alpha 风险 | 行业平均 |

PM 必须明确推荐其中之一，并解释理由。

---

## 输入资料

### Research Manager 的 thesis（核心输入）
{research_plan}

### 股票画像（决定决策风格 + 报告使用权重 + Time Stop / Entry 节奏）
{stock_profile if stock_profile else "（未提供）"}

### 共识快照（用于 entry timing 判断）
{consensus_snapshot if consensus_snapshot else "（未提供）"}

### 4 个 analyst 原始报告（PM 独享，用于校验 RM thesis + 操作细节）

[置信度:高] Company fundamentals report:
{fundamentals_report}

[置信度:中高] Market research report:
{market_report}

[置信度:中] Latest world affairs news:
{news_report}

[置信度:中低] Social media sentiment report:
{sentiment_report}

### 风险团队辩论记录
{history}

### 历史教训（BM25 检索出的最相关 3 条，**必须在第九步自检表里逐条对照**）
{past_memory_str if past_memory_str else "（本次未检索到相关教训，第九步自检表填'本次无相关历史教训'即可）"}

---

## 关键原则

- **事实校验**：风控辩论各方对同一数据有不同解读时，以 RM 引用数据为准
- **PE/EPS 估值**：以分析师报告中"系统计算"值为准
- **数学优先**：R-multiple、Conviction、情景概率必须显式计算，禁止凭感觉
- **完整性**：所有 11 个章节必须输出，HOLD/UNDERWEIGHT/SELL 时 Entry 字段填 "—" 但其他不准省略

{RISK_DEBATE_PHRASING_RULES}

**重要**：请用中文撰写。评级关键词、股票代码、交易术语（Action/Size/R/TP/SL/Time Stop）保留英文但带中文注释。

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        # 绑定 PM 计算工具，让 LLM 调工具算 R-multiple / Conviction / 4 情景 E
        llm_with_tools = llm.bind_tools(PM_TOOLS)
        response = _pm_tool_loop(llm_with_tools, [HumanMessage(content=prompt)])

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return portfolio_manager_node
