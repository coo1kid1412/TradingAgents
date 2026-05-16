
from tradingagents.agents.utils.agent_utils import build_instrument_context, RISK_DEBATE_PHRASING_RULES


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"], state.get("company_name", ""))
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        consensus_snapshot = state.get("consensus_snapshot", "")

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{fundamentals_report}\n\n{market_research_report}\n\n{news_report}\n\n{sentiment_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=3)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""【语言要求】你必须使用中文撰写以下所有分析内容和回复。评级关键词（Buy/Overweight/Hold/Underweight/Sell）和股票代码可保留英文。

你是**投资研究总监（Head of Research）**。

## 你的角色边界（必读）

你**只负责出 thesis**：
- 综合多空论据 → 量化评分 → 评级
- 给出目标价区间 P_up / P_dn
- 列出证伪触发器
- 为风控团队指出审查重点

你**不负责**：
- 建仓价位、止损位、仓位比例（这些是 Portfolio Manager 的职责）
- 执行节奏、流动性评估（这些是 PM 的职责）
- 操作动作表（这些是 PM 的职责）

**禁止越权**：你的报告里不准出现"首笔建仓 X 元"、"止损 Y 元"、"建议仓位 Z%"这类执行细节。如果你想表达"赔率不利"，请用 P_up/P_dn 和 R 数值表达，让 PM 自己决定怎么操作。

{instrument_context}

---

## 决策流程（必须严格按顺序完成）

### 第一步：吸收市场共识快照

共识识别官已经提炼好了"当前股价隐含的市场预期"。直接引用，不要重做：

{consensus_snapshot if consensus_snapshot else "（共识快照缺失，请基于 sentiment + news 自行简要识别共识方向）"}

**评分时的对称校准规则**（强制应用）：
- 任何与共识方向同向的论据（多头或空头）→ 该条得分**封顶 6 分**（已被定价，不构成 alpha）
- 任何反共识论据（指出共识忽视的点）→ 该条得分**下限抬高 1 分**（contrarian 视角是高 alpha 来源）
- 反共识但有扎实论据 → 允许给到 9-10 分

**输出**：
> 共识方向：[偏多/偏空/分歧]；共识强度：[强/中/弱]。本次评分按对称校准规则应用。

---

### 第二步：反驳质量评估（评分前必做）

在打分前，先列出：
- **多头被有效反驳的论据**（最多 2 条）：哪条多头论据被空头用数据驳倒？驳倒方式（引用错误数据 / 提出反例数据 / 揭示前提失效）
- **空头被有效反驳的论据**（最多 2 条）：同上

**反驳影响评分**：
- 被有效反驳的论据：得分 **-1 ~ -2 分**（驳倒越彻底，扣分越多）
- 被指出"引用错误数据"的论据：**直接 0 分剔除**
- 未被反驳或反驳无力的论据：得分不变

---

### 第三步：量化评分（按证据类型 + Hard Data 加权）

⚠️ **核心规则：禁止使用算术平均，禁止按论据排名分配权重。必须按"证据类型 × Hard Data"双因子加权。**

**权重计算**：

| 证据类型 | 基础权重 | 识别标准 |
|---------|---------|---------|
| Hard fact | 3.0 | 财报披露数字、监管文件、已公告事件——可在原始报告中找到具体数值/原文，时间已发生 |
| Catalyst | 2.0 | 已宣布但未兑现的事件，含明确时间窗口（例如"6 月新品发布"、"Q3 财报指引"）|
| 估值类比 / 趋势外推 | 1.0 | 例如"PE 低于行业平均"、"技术面突破" |
| 情绪 / 叙事 / 主观判断 | 0.5 | 例如"市场关注度提升"、"投资者情绪转暖" |

**Hard Data 修正**（叠加于证据类型权重）：
- 论据 Hard Data = yes：权重不变
- 论据 Hard Data = no：权重 × 0.5（缺乏数值支撑的论据可信度对折）
- 论据未标注 Hard Data：按 no 处理

例：Hard fact + Hard Data=yes → 权重 3.0；Hard fact + Hard Data=no → 权重 1.5；Catalyst + Hard Data=yes → 权重 2.0

**额外规则**：
- 得分 ≤ 3 分的论据**直接剔除**，不计入加权平均
- 同一证据若被双方都引用但解读相反（如 PE 数值矛盾）→ 引用错误数据的一方该条**直接 0 分剔除**
- 共识校准（第一步）+ 反驳影响（第二步）必须显式应用到每条得分

#### 多头论据明细

| 序号 | 论据 | 立场 | 证据类型 | Hard Data | 原始得分 | 共识校准 | 反驳调整 | 最终得分 | 权重 |
|------|------|------|---------|-----------|---------|---------|---------|---------|------|
| 1 | xxx | 共识/反共识/中性 | __ | yes/no | x | -0/+1/封顶6 | -0/-1/-2 | x | __ |
| 2 | ... | ... | ... | ... | ... | ... | ... | ... | ... |

**多头剔除论据**：列出被剔除的低质论据（得分 ≤3 或数据错误），并说明剔除原因。
**权重之和 W_bull = ___**
**多头总分 Bull Score = Σ(最终得分_i × 权重_i) / W_bull = ___（保留 2 位小数，必须展开乘法过程）**

#### 空头论据明细

| 序号 | 论据 | 立场 | 证据类型 | Hard Data | 原始得分 | 共识校准 | 反驳调整 | 最终得分 | 权重 |
|------|------|------|---------|-----------|---------|---------|---------|---------|------|
| 1 | xxx | ... | ... | ... | ... | ... | ... | ... | ... |

**空头剔除论据**：列出被剔除的低质论据。
**权重之和 W_bear = ___**
**空头总分 Bear Score = Σ(最终得分_i × 权重_i) / W_bear = ___（保留 2 位小数，必须展开乘法过程）**

#### 得分差

**d = Bull Score - Bear Score = ___（保留 2 位小数）**

---

**评分标准**：
- 9-10 分：论据有确凿的数据/事件支撑，逻辑严密，短期内大概率兑现
- 7-8 分：论据较有说服力，有明确的催化剂或数据支持
- 5-6 分：论据有一定道理，但缺乏关键数据或催化剂
- 4 分：论据较弱，存在明显逻辑漏洞（处于剔除边缘）
- ≤3 分：论据站不住脚（**直接剔除**）

## ⚠️ 数值类论据的评分注意事项
- **PE 估值类论据**：评定时以分析师报告中「系统计算」的 PE 值为准。如果辩论双方引用了不同的 PE 值，引用错误数据的一方该条论据**直接 0 分剔除**
- **估值指标类论据**：检查论据中引用的 EPS、ROE、毛利率等是否与原始数据一致。如果发现明显计算错误，该论据得分不应超过 4 分
- **交叉校验**：对于 PE 相关的论据，检查 PE × EPS 是否约等于报告中的收盘价。偏差超过 15% 的，在评分理由中说明

---

### 第四步：根据得分差决定初评评级 R0（5 档制）

按以下阈值选择初评评级 R0：

| 得分差 d | 评级 | 操作含义 |
|---------|------|---------|
| d > 1.5 | **BUY** | 信号明确，方向显著偏多 |
| 0.5 ≤ d ≤ 1.5 | **OVERWEIGHT** | 方向偏多但证据不足以重仓 |
| -0.5 < d < 0.5 | **HOLD** | 多空均衡，无明确方向 |
| -1.5 ≤ d ≤ -0.5 | **UNDERWEIGHT** | 方向偏空但证据不足以清仓 |
| d < -1.5 | **SELL** | 信号明确，方向显著偏空 |

**边界归属**：d=0.5、d=1.5 都是 OVERWEIGHT；d=-0.5、d=-1.5 都是 UNDERWEIGHT。

记录 **R0 = ___**

### 第五步：赔率与胜率校验（强制执行）

加权得分差 d 仅表达"哪边论据更强"，但未表达"对的时候能赚多少 / 错的时候能亏多少"。

| 项目 | 数值 | 依据 |
|------|------|------|
| 基准价 P_0（当前价/最近收盘）| ___ | 引用市场报告 |
| 上行目标价 P_up | ___ | 引用关键多头论据，说明定价方法 |
| 下行风险价 P_dn | ___ | 引用关键空头论据，说明定价方法 |
| 上行幅度 U = (P_up - P_0) / P_0 | ___% | |
| 下行幅度 D = (P_0 - P_dn) / P_0 | ___% | |
| 赔率 R = U / D | ___ | |
| 主观胜率 p | ___ | 须基于 d 校准：d>1.5 时 p≥0.6；0.5≤d≤1.5 时 0.5≤p≤0.6；d 绝对值<0.5 时 p≈0.5；-1.5≤d≤-0.5 时 0.4≤p≤0.5；d<-1.5 时 p≤0.4 |
| 期望收益 E = p × U − (1−p) × D | ___% | |

**评级修正规则（R1，对称升降档）**：

- 若 R0 ∈ {{BUY, OVERWEIGHT}} 但 (E ≤ 0 或 R < 1.5) → R1 = R0 **降一档**
- 若 R0 ∈ {{SELL, UNDERWEIGHT}} 但 (E ≥ 0 或 R > 0.67) → R1 = R0 **升一档**
- 若 R0 = OVERWEIGHT 且 (R ≥ 2.0 且 E > 8%) → R1 = **BUY**（赔率极优可升档）
- 若 R0 = UNDERWEIGHT 且 (R ≤ 0.5 且 E < -8%) → R1 = **SELL**（赔率极差可升档）
- 若 R0 = HOLD 且 (R ≥ 3 且 E > 5%) → R1 = OVERWEIGHT（极端非对称机会，破例升档）
- 若 R0 = HOLD 且 (R ≤ 0.33 且 E < -5%) → R1 = UNDERWEIGHT（极端非对称风险，破例降档）
- 否则 R1 = R0

**记录 R1 = ___**，并显式写出修正路径。

### 第六步：关键证据鲁棒性自检（Anchor 检验，Top-2）

假设辩论中**最关键的 2 条 hard fact 同时被证伪**，重算 d'。这是测多变量鲁棒性。

1. **识别 anchor evidence**（按 R1 方向）：
   - 若 R1 ∈ {{BUY, OVERWEIGHT}} → 找多头侧 **得分 × 权重 最高的 2 条 hard fact**
   - 若 R1 ∈ {{SELL, UNDERWEIGHT}} → 找空头侧同理
   - 若 R1 = HOLD → 跳过本步，R2 = R1
   - 若该侧 hard fact 不足 2 条（评级建立在 catalyst/估值上）→ 强制 R2 = HOLD（无硬证据支撑的方向性下注不可接受）

2. **假设两条 anchor 同时失效**：将这 2 条得分置为 0，权重保留，重新计算该侧总分。

3. **重新计算 d'**，按第四步阈值表判断是否跨档。

4. **决策规则（对称升降档）**：
   - 若 d' 跨过 2 档以上 → R2 = HOLD
   - 若 d' 跨过 1 档（方向减弱）→ R2 = R1 **降一档**
   - 若 d' 未跨档 → R2 = R1（决策对 anchor 不敏感）
   - 若 d' 跨档**朝相反方向加强**（多头 anchor 失效后反而 d' 变得更负）→ 必定 R2 = HOLD（论据矛盾）
   - 若空头侧 anchor 失效后 d' 进一步偏多（罕见）→ R2 = R1 **升一档**

**输出表格**：

| 项目 | 数值 |
|------|------|
| R1 | __ |
| Anchor 1 描述 | __ |
| Anchor 1 得分 / 权重 | __ / __ |
| Anchor 2 描述 | __ |
| Anchor 2 得分 / 权重 | __ / __ |
| 该侧原总分 | __ |
| 该侧假设失效后总分 | __ |
| 原 d / 假设失效后 d' | __ / __ |
| 跨档数 | __ |
| **最终评级 R2** | __ |

### 第七步：制定 Thesis（最终输出）

**注意：你只输出 thesis 元素，不要给执行建议。**

#### 7.1 评级与置信度

| 字段 | 内容 |
|------|------|
| 最终评级 R2 | __ |
| 评级置信度 | 高 / 中 / 低（基于 |d|：>2.0 高，1.0-2.0 中，<1.0 低）|
| 修正路径 | R0 → R1 → R2 的演化（一句话）|

#### 7.2 目标价区间

| 字段 | 数值 |
|------|------|
| 当前价 P_0 | __ 元 |
| 上行目标价 P_up | __ 元（含定价方法） |
| 下行风险价 P_dn | __ 元（含定价方法） |
| 赔率 R | __ |
| 期望收益 E | __ % |

#### 7.3 核心 Thesis（3-5 条最有说服力的论据）

直接列出辩论中**最强、未被剔除**的 hard fact / catalyst。每条 1-2 句话，注明立场（共识/反共识）。

#### 7.4 证伪触发器（必须给出 3 条，每条满足"具体数据点 + 阈值 + 时间窗口 + 触发动作"四要素）

| # | 触发条件（必须可观测）| 时间窗口 | 触发动作 |
|---|---------------------|---------|---------|
| 1 | 例：Q2 营收同比增速 < 12% | 下次财报披露后立即检查 | 评级降至 HOLD |
| 2 | 例：行业政策出现 X 类负面表述 | 任意时点 | 评级降至 UNDERWEIGHT |
| 3 | 例：当前价跌破 ___ 元并伴随成交放量 2 倍以上 | 5 个交易日内 | 评级降至 SELL |

**禁止泛化表述**："基本面恶化"、"宏观转弱"、"市场情绪变差"、"竞争加剧" 都不可接受——必须能让看到这条记录的人在 6 个月后**机械地判断是否被触发**。

#### 7.5 反面风险（一句话）

当前 thesis 最大的反面风险是什么？若错，最可能错在哪里？

### 第八步：风控审查指引（必须输出）

风控团队将基于你的 thesis 进行维度化审查。指出审查重点：

1. **未决问题**：Bull/Bear 辩论中未达成共识的 2-3 个核心分歧点
   - 分歧点1：[描述] — 多头认为[X]，空头认为[Y]
   - 分歧点2：...
2. **风控审查建议**：
   - 评级偏多 + 流动性疑虑 → 重点审查执行可行性
   - 评级依赖未来催化剂 → 重点审查事件风险时间窗口
   - |d| < 0.7 → 重点审查极端情景尾部风险
   - 赔率 R 接近 1.5 或 0.67 → 重点审查目标价假设鲁棒性
   - Anchor 检验跨档或无 hard fact 降档 → 重点审查证据链完整性

---

## 历史教训
\"{past_memory_str}\"

## 原始分析师报告（用于交叉校验辩论中的数据引用）

[置信度:高] Company fundamentals report: {fundamentals_report}
[置信度:中高] Market research report: {market_research_report}
[置信度:中] Latest world affairs news: {news_report}
[置信度:中低] Social media sentiment report: {sentiment_report}

**交叉校验要求**：在评分时，如果 Bull/Bear 引用的数据与原始报告不一致（如 PE 数值矛盾、事实错误），必须按"数值类论据评分注意事项"扣减或剔除。置信度高的报告（基本面、市场技术面）优先采信。

## 辩论记录
{history}

---

{RISK_DEBATE_PHRASING_RULES}

**重要：请用中文撰写你的 thesis 报告。** 评级关键词（Buy/Overweight/Hold/Underweight/Sell）和股票代码请保留英文原文。

**最后提醒：你只出 thesis，不出执行细节。建仓价、止损价、仓位比例由 PM 决定。**
"""
        response = llm.invoke(prompt)

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
