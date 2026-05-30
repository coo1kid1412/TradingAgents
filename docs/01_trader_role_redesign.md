# 优化 01：Trader 角色重构 —— 从"二次决策者"改为"执行专家"

## 1. 背景与现状问题

### 1.1 当前流程
图工作流（`tradingagents/graph/setup.py`）目前的链路是：

```
4 个分析师 → Bull/Bear 辩论 → Research Manager → Trader → 风控三方辩论 → Portfolio Manager → END
```

其中 **Research Manager**、**Trader**、**Portfolio Manager** 三个节点都被要求"判断方向、给出评级、制定交易计划"，职责高度重叠。

### 1.2 真实报告中的证据

以 `reports/300857_协创数据_20260509_211340/` 为例：

- `2_research/manager.md` 给出多头 5/10、空头 6/10、得分差 -1，最终评级 **HOLD**
- `3_trading/trader.md` 给出多头 5/10、空头 6/10、得分差 -1，最终评级 **HOLD**
- `5_portfolio/decision.md` 给出多头 5/10、空头 6/10、得分差 -1，最终评级 **HOLD**

三份报告核心结论几乎一字不差，连"风险收益比向上 23%/向下 19%"这样的具体数字都重复。**Trader 没有提供任何独有视角。**

另一个反例是 `reports/002460_赣锋锂业_20260509_171238/`：

- Research Manager 判 HOLD
- Trader 判 BUY（弱看多，建议 75-78 元试探建仓）
- Portfolio Manager 直接采纳了 Research Manager 的 HOLD，对 Trader 的分歧仅一笔带过

这说明 Trader 当前要么和 Research Manager 完全重复，要么被 Portfolio Manager 直接覆盖。无论哪种情况，它的方向判断都没有产生有效价值。

### 1.3 prompt 层面的问题

`tradingagents/agents/trader/trader.py` 的 prompt（"决策流程"小节）明确要求 Trader：

- 第一步：判断市场方向（看多/看空）
- 第二步：确定操作建议（BUY/HOLD/SELL）
- 第三步：交易计划

这把 Trader 定位成"再决策一次"，与 Research Manager 高度同质化。

---

## 2. 目标

把 Trader 从"二次方向决策者"改造成"执行专家"：

- **不再给方向**。Trader 的输出直接采纳 Research Manager 的方向判断
- **专注执行细节**：入场切片、单笔仓位、止损位合理性、时间窗口、流动性约束
- **对 Portfolio Manager 提供独立增量价值**：执行可行性评估、与建议方向冲突的执行风险

完成后的 Trader 报告应该与 Research Manager 报告**完全不重叠**，但又是 Portfolio Manager 做最终决策时不可或缺的一份输入。

---

## 3. 改动范围

### 3.1 必须修改

- `tradingagents/agents/trader/trader.py` —— 重写 system prompt，调整输出结构
- `tradingagents/agents/managers/portfolio_manager.py` —— prompt 中调整对 trader_plan 的描述与使用方式

### 3.2 不修改（明确划出）

- `tradingagents/graph/setup.py`：图节点和边保持不变，Trader 仍在 Research Manager 之后、风控辩论之前
- `tradingagents/graph/conditional_logic.py`：路由逻辑不变
- `tradingagents/graph/signal_processing.py`：评级解析逻辑不变（Trader 输出仍然以 `FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**` 结尾，但该方向直接镜像 Research Manager）
- `tradingagents/agents/managers/research_manager.py`：本次不动
- `tradingagents/agents/utils/memory.py` 与 `trader_memory`：本次不动（trader 仍可读取自己的反思记忆，但记忆内容由 reflector 在事后写入，与本次改动无关）

---

## 4. 详细变更

### 4.1 `tradingagents/agents/trader/trader.py`

#### 删除
- 删除"第一步：判断市场方向"小节
- 删除"第二步：确定操作建议"小节及其映射规则
- 删除现有的"⚠️ 数值类指标审查规范"小节（数据校验是 fundamentals 层的职责，将由优化 03 处理；本次先从 trader 移除）

#### 保留
- 中文输出要求
- `FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**` 结尾格式（用于 signal_processing 兼容）
- 历史教训（past_memory_str）的注入

#### 新增/重写
重写 system prompt 的主体，让 Trader 的角色变成：

> 你是一名交易执行专家。Research Manager 已经给出了方向判断和评级（见 investment_plan）。你不重新判断方向，你的工作是**评估并细化执行方案**。

要求 Trader 输出以下章节（中文）：

1. **方向确认**：一句话复述 Research Manager 的方向和评级，明确表示"采纳"
2. **入场策略**：
   - 一次性建仓 vs 分批建仓（若分批，给出每批的触发条件——价格信号、技术信号、时间信号）
   - 单笔上限、总仓位上限
3. **止损位合理性评估**：
   - Research Manager 给的止损位是否合理？参考 ATR、最近支撑位、波动率
   - 给出最终建议的硬止损与时间止损
4. **流动性与时间窗口**：
   - 该标的近期日均成交额，建议仓位是否会造成滑点
   - 是否临近财报披露窗口、是否避开
5. **执行风险**：
   - 即使 Research Manager 的方向正确，执行层可能出问题的点（流动性枯竭、停牌、涨跌停板限制等）
6. **结尾**：保留 `FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`，方向必须与 Research Manager 一致（取自 `investment_plan` 中的评级）

#### 镜像方向的实现要点

Trader 不重新判断方向，但仍需在 prompt 末尾保留 `FINAL TRANSACTION PROPOSAL` 格式以保持 signal_processing 兼容。在 prompt 中明确指示：

> 在 investment_plan 中查找 Research Manager 给出的评级关键词（强烈买入/买入/增持/持有/中性/减持/卖出/强烈卖出 或对应英文），将其折叠到 BUY/HOLD/SELL 三档：
> - 买入类（STRONG_BUY/BUY/OVERWEIGHT）→ BUY
> - 持有类（HOLD）→ HOLD
> - 卖出类（UNDERWEIGHT/SELL/STRONG_SELL）→ SELL
> 然后输出 `FINAL TRANSACTION PROPOSAL: **<对应词>**`

这样不依赖正则解析 Research Manager 输出，由 LLM 在 prompt 内完成识别。

### 4.2 `tradingagents/agents/managers/portfolio_manager.py`

#### 修改 prompt 中对 `trader_plan` 的描述

当前 prompt 把 `trader_plan` 当作"交易员的初步方案"。改造后应明确说明：

> Trader 提供的是**执行方案**（入场切片、止损位、流动性评估），不再重新判断方向。Trader 的方向已与 Research Manager 一致。

#### 在决策流程中加入"执行可行性"环节

在第二步（确定评级强度）和第三步（输出报告）之间增加一句：

> **执行可行性核查**：参考 Trader 的执行方案，确认你的最终建议在执行层可行。若 Trader 标记的执行风险（流动性、滑点、停牌窗口等）足以影响决策，需在最终报告中体现。

#### 不要求新增分歧记录小节

"显式列出与 Trader 的分歧"是优化 04 的范畴，本次不实现。

---

## 5. 验收标准

跑 3 只标的（建议：A 股 1 只新跑、港股 1 只、美股 1 只），生成新报告后逐项核对：

### 5.1 Trader 输出（trader.md）

- [ ] **不包含** "第一步：判断市场方向" / "市场方向：看多/看空" 等表述
- [ ] **不包含** 多空得分（5/10、6/10 之类）的二次打分
- [ ] **包含** 以下章节标题（或语义等价的中文表述）：方向确认、入场策略、止损位合理性、流动性与时间窗口、执行风险
- [ ] 文末保留 `FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**` 格式
- [ ] 该 FINAL TRANSACTION PROPOSAL 的方向，与同一份 `2_research/manager.md` 的最终评级一致（按 BUY/HOLD/SELL 三档折叠）

### 5.2 Portfolio Manager 输出（decision.md）

- [ ] decision.md 的最终评级，与改造前的同标的报告（如有历史报告可对比）相比，没有出现解析失败或评级缺失
- [ ] 当 Trader 在执行风险章节标记了具体问题时，decision.md 的"风险提示"或"执行摘要"中能找到对应表述（不强制完整复述，但需可见 Trader 的影响）

### 5.3 系统层面

- [ ] CLI 完整跑通，`reports/<ticker>/<timestamp>/` 目录结构与现状一致（仍有 `1_analysts` / `2_research` / `3_trading` / `4_risk` / `5_portfolio` / `complete_report.md`）
- [ ] `signal_processing.SignalProcessor.process_signal` 对新版 trader.md 和 decision.md 都能成功提取出有效评级（不是 fallback 的默认 HOLD）
- [ ] 跑 3 只标的时，没有出现 LLM 调用失败或 graph 路由错误

---

## 6. 测试计划

### 6.1 标的选择
- A 股个股 1 只：建议从已有报告中挑一只 Research Manager 给出非 HOLD 评级的（便于验证 Trader 镜像方向是否正确）
- 港股 1 只：从 `reports/2729.HK_GALAXIS TECH_*` 或 `reports/02616.HK_*` 中选最近一份对应的标的重新跑
- ETF 1 只：从 `reports/159326_*` 选一只

### 6.2 对比方法
对每只标的：
1. 保留现有 `reports/<ticker>/<旧时间戳>/` 作为 baseline
2. 跑新版生成 `reports/<ticker>/<新时间戳>/`
3. 重点对比 `3_trading/trader.md` 新旧两版：
   - 旧版应包含"第一步：判断市场方向"
   - 新版应替换为"方向确认 / 入场策略 / 止损位合理性 / 流动性与时间窗口 / 执行风险"
4. 对比 `5_portfolio/decision.md`：评级保持一致或仅在执行细节描述上有差异

### 6.3 必跑的边界情况
- 一只 Research Manager 判 BUY 的标的：验证 Trader 是否输出 `FINAL TRANSACTION PROPOSAL: **BUY**` 而非 HOLD
- 一只 Research Manager 判 HOLD 的标的：验证 Trader 不试图改写为 BUY/SELL

---

## 7. 风险与回滚

### 7.1 主要风险

- **Trader 仍偷偷给方向**：LLM 可能在执行讨论中夹带方向判断。通过验收标准 5.1 的前两项检测
- **方向镜像不准**：LLM 在折叠 7 级评级到 3 级时出错。通过验收标准 5.1 最后一项检测
- **Trader 输出过于干瘪**：删了方向决策后，如果执行细节也写不深入，trader.md 可能变成空洞文档。需在测试时人工抽查至少 1 份新版 trader.md，确认执行细节有实质内容

### 7.2 回滚

- 所有改动应在**单个 commit** 中完成，便于 `git revert` 一键回退
- commit message 建议：`refactor(trader): 重构为执行专家角色，移除二次方向决策`
- 如果验收不通过，返工时也保持单 commit 原则（amend 或新 commit 二选一，由用户决定）

---

## 8. 不在本次范围

以下议题与本次改造相邻，但**本次不动**，避免改动面失控：

- **HOLD 阈值放松**（优化 02）：Research Manager 的 HOLD 判定逻辑本次不改
- **fundamentals 数据校验**（优化 03）：Trader prompt 中现有的"PE 数据审查"段落本次只做删除，不在其他位置补上
- **decision.md 强制分歧记录**（优化 04）：本次只在 Portfolio Manager prompt 中说明 Trader 不再给方向，不要求新增"分歧"小节
- **删除 Trader 节点**：不删，保留节点结构。删除是一个更激进的方案，留待优化 02-04 全部完成后视效果再评估
- **trader_memory 的反思机制**：保持现状，由 `reflect_trader` 在事后写入，本次不动
- **风控三方辩论的内容**：Aggressive/Conservative/Neutral 的 prompt 本次不改，他们仍可在辩论中提及方向

---

## 9. 实施完成后

Qoder 完成后请回传：
1. 修改的文件列表（应只有 `trader.py` 和 `portfolio_manager.py`）
2. 实际跑过的 3 只标的的 ticker + 时间戳
3. 任意 1 份新版 trader.md 的全文（粘贴在回执里）

由 Claude 复核验收标准后，更新 `docs/ROADMAP.md` 中本项状态为"已完成"，并开始撰写优化 02 的规格。
