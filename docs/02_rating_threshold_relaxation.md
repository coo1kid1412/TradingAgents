# 优化 02：放松 HOLD 阈值，启用 5 档评级 + 敏感性自检

## 1. 背景与现状问题

### 1.1 当前评级机制

`tradingagents/agents/managers/research_manager.py` 第 50-58 行写死：

```
- 多头得分 - 空头得分 > 1 → Buy
- 空头得分 - 多头得分 > 1 → Sell
- 得分差 ≤ 1 → Hold
```

`tradingagents/agents/managers/portfolio_manager.py` 第 50-59 行虽然在表格里列了 Overweight/Underweight，但实操描述是"如果研究经理给出的是 Hold，你应选择 Hold"——**OVERWEIGHT/UNDERWEIGHT 实际上从未被生效使用**。

### 1.2 真实报告中的问题

历史报告（如 `reports/300857_协创数据_*` 的 9 次跑批）几乎全是 HOLD。原因：

1. **阈值过粘**：得分差 1 分以内全部归 HOLD，但 LLM 给同一组论据打 5/10 还是 6/10 本身就是 token 概率级别的随机扰动
2. **缺乏弱信号档位**：方向有倾向但证据不足以重仓的状态（最常见的真实投资场景）被强行压成"无观点"
3. **得分本身脆弱**：现有机制让 LLM 对每条论据 1-10 打分后求平均，没有对评分稳健性做任何校验

### 1.3 行业基准

A 股研报和卖方机构通用的 5 档评级：**买入 / 增持 / 中性 / 减持 / 卖出**（对应 BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL）。这套体系几十年来就是为了表达"方向有倾向但置信度不足"的中间态。

`tradingagents/graph/signal_processing.py` 的 `_CN_TO_EN` 映射已经包含完整 7 级映射，**基础设施齐备，缺的只是让 prompt 实际产出这些档位**。

---

## 2. 目标

完成后达到三条同时成立：

1. **5 档评级生效**：Research Manager 与 Portfolio Manager 的输出在多只标的上能够实际产出 OVERWEIGHT/UNDERWEIGHT，而不是清一色 HOLD
2. **敏感性自检常态化**：Research Manager 在给出非 HOLD 评级前必须做一次"如果最有力的两条论据各弱 1 分，结论是否翻转"的自检；翻转则降一档
3. **Portfolio Manager 与 Research Manager 解耦但克制**：PM 默认采纳 RM 的评级，**仅当执行层风险 dominant 时**可调整 ±1 档，**不允许跨方向翻转**（Buy 不能变 Sell）

---

## 3. 改动范围

### 3.1 必须修改

- `tradingagents/agents/managers/research_manager.py` —— 评级阈值改为 5 档；新增敏感性自检步骤；明确每档的操作含义
- `tradingagents/agents/managers/portfolio_manager.py` —— 改为"默认采纳 RM 评级，可调 ±1 档但不能翻方向"的逻辑

### 3.2 不修改（明确划出）

- `tradingagents/graph/signal_processing.py` —— **不动**。继续保留 7 级解析能力，对 LLM 偶尔产出的 STRONG_BUY/STRONG_SELL 优雅降级（输入宽容、输出克制）
- `tradingagents/agents/trader/trader.py` —— **不动**。优化 01 已写好的折叠规则（OVERWEIGHT→BUY、UNDERWEIGHT→SELL）天然兼容 5 档输出
- `tradingagents/graph/setup.py` / `conditional_logic.py` —— **不动**
- 风控三方辩论 —— **不动**

---

## 4. 详细变更

### 4.1 `tradingagents/agents/managers/research_manager.py`

#### 4.1.1 替换"第二步：根据得分差决定方向"小节

旧的二档阈值：

```
- 多头得分 - 空头得分 > 1 → Buy
- 空头得分 - 多头得分 > 1 → Sell
- 得分差 ≤ 1 → Hold
```

替换为五档阈值（**只允许产出这 5 档**，不允许 STRONG_BUY/STRONG_SELL）：

| 得分差 d = 多头得分 - 空头得分 | 评级 | 中文 | 操作含义 |
|------|------|------|---------|
| d > 1 | **BUY** | 买入 | 信号明确，建议建仓或加仓至目标比例 |
| 0.5 ≤ d ≤ 1 | **OVERWEIGHT** | 增持 | 方向偏多但证据不足以重仓，建议轻仓试探或保留现金继续观察 |
| -0.5 < d < 0.5 | **HOLD** | 中性 | 多空均衡，维持现有仓位，不主动调整 |
| -1 ≤ d ≤ -0.5 | **UNDERWEIGHT** | 减持 | 方向偏空但证据不足以清仓，建议部分止盈或减仓 |
| d < -1 | **SELL** | 卖出 | 信号明确，建议清仓离场 |

**关键**：每档的"操作含义"必须在 prompt 里写清楚，否则 LLM 会把 OVERWEIGHT 当作"弱 BUY" 用，与 BUY 没差别。

#### 4.1.2 在第二步与第三步之间插入新的"第二步 B：敏感性自检"

在 prompt 中加入以下指令（用中文，下面是设计意图，Qoder 落地时按此 paraphrase 即可）：

> ### 第二步 B：敏感性自检（强制执行）
> 
> 在给出非 HOLD 评级前，你必须做一次评分稳健性自检：
> 
> 1. 找出你打分中**最有说服力的 2 条论据**（无论多头还是空头）
> 2. 假设这 2 条论据各自被弱化 1 分（例如从 7 降到 6），重新计算得分差
> 3. 如果重新计算后的得分差**跨过了相邻档位的边界**（例如从 d=1.2 降到 d=0.8，从 BUY 区进入 OVERWEIGHT 区），说明你的评级建立在 LLM 评分的边缘扰动上
> 4. 此时**强制降一档**：BUY → OVERWEIGHT、OVERWEIGHT → HOLD、UNDERWEIGHT → HOLD、SELL → UNDERWEIGHT
> 5. HOLD 不需要降档
> 
> 输出时必须显式包含一个"敏感性自检"小节，写明：原始得分差 / 假设弱化后的得分差 / 是否触发降档 / 最终评级。

#### 4.1.3 修改"第三步：制定投资计划"

把原来的 `Buy / Sell / Hold` 改为 `Buy / Overweight / Hold / Underweight / Sell`，并在每档的"行动方案"小节示例里区分：

- **BUY** 的行动方案：建仓比例 50-100%、明确入场区间
- **OVERWEIGHT** 的行动方案：建仓比例 ≤30%、保留 ≥50% 现金等待加仓信号
- **HOLD** 的行动方案：维持现有仓位，给出后续观察的触发条件
- **UNDERWEIGHT** 的行动方案：减仓 30-50%、保留底仓观察
- **SELL** 的行动方案：清仓或减仓至 ≤10%

#### 4.1.4 保留不动

- 评分标准（1-10 分的语义描述）
- 数值类论据评分注意事项（PE 一致性等）
- 历史教训注入逻辑

### 4.2 `tradingagents/agents/managers/portfolio_manager.py`

#### 4.2.1 替换"第二步：确定评级强度"小节

原表格保留 5 档（去掉 Buy 与 Strong Buy 的区分，全部归到 Buy）。

#### 4.2.2 替换"重要约束"段落

把原来"如果研究经理给出的是 Hold，你应选择 Hold"这种粘 RM 的措辞，改为：

> ### Portfolio Manager 评级决策规则
> 
> 1. **默认评级**：直接采纳 Research Manager 的评级
> 2. **允许的调整**：仅当**执行层因素**足以改变操作建议时，可在 RM 评级基础上调整 **±1 档**
>    - 可调整的合理理由：流动性枯竭、临近重大事件窗口（财报/解禁/政策细则）、Trader 标记的关键执行风险
>    - 不可调整的理由：对 bull/bear 辩论的二次评判（这不是你的职责）、对宏观环境的额外判断（应在风控辩论中体现）
> 3. **禁止跨方向翻转**：BUY/OVERWEIGHT 不能变成 UNDERWEIGHT/SELL，反之亦然。最大调整范围：
>    - BUY ↔ OVERWEIGHT（信号强弱微调）
>    - UNDERWEIGHT ↔ SELL（信号强弱微调）
>    - OVERWEIGHT ↔ HOLD（执行风险压制方向倾向）
>    - UNDERWEIGHT ↔ HOLD（执行风险压制方向倾向）
> 4. **调整必须留痕**：若最终评级与 RM 不同，必须在报告中包含"评级调整说明"小节，明确说明调整方向（升档/降档）+ 触发理由（哪个执行层因素）

#### 4.2.3 在第三步输出报告中明确

```
2. **评级**：Buy / Overweight / Hold / Underweight / Sell
   - 若与 Research Manager 评级一致：直接给出
   - 若不一致：必须附"评级调整说明"小节
```

#### 4.2.4 保留不动

- 第一步的方向判断逻辑
- 执行可行性核查（优化 01 加入的）
- 历史教训和风控辩论注入逻辑

---

## 5. 验收标准

跑 5 只标的（建议覆盖：A 股个股 2 只、港股 1 只、美股 1 只、ETF 1 只）。**特别要求**：尽量挑选 bull/bear 辩论中双方论据接近的标的（比如历史上一直跑出 HOLD 的），以验证新阈值能否产出 OVERWEIGHT/UNDERWEIGHT。

### 5.1 Research Manager 输出（manager.md）

- [ ] 5 只标的中，**至少 2 只**的最终评级是 OVERWEIGHT / UNDERWEIGHT 之一（不是 100% HOLD 或 100% BUY/SELL）
- [ ] 所有非 HOLD 评级的 manager.md 都包含"敏感性自检"小节，且小节内容完整（原始得分差 / 弱化后得分差 / 是否触发降档 / 最终评级）
- [ ] 没有任何一份 manager.md 输出 STRONG_BUY 或 STRONG_SELL
- [ ] OVERWEIGHT 评级对应的"行动方案"建仓比例 ≤30%、保留现金 ≥50%（按 prompt 要求）；UNDERWEIGHT 对应减仓 30-50%

### 5.2 Portfolio Manager 输出（decision.md）

- [ ] 所有 decision.md 的最终评级是 5 档之一（BUY/OVERWEIGHT/HOLD/UNDERWEIGHT/SELL），无 STRONG_*
- [ ] 当 PM 评级与 RM 评级不同时，**必须**包含"评级调整说明"小节，理由必须引用执行层因素（流动性、事件窗口、Trader 标记的风险），不能引用对 bull/bear 辩论的重新判断
- [ ] 无任何 PM 评级跨方向翻转 RM 评级（不允许 BUY → SELL 这种变化）

### 5.3 系统层面

- [ ] CLI 完整跑通 5 只标的
- [ ] `signal_processing.SignalProcessor.process_signal` 对所有 5 档评级都能成功提取（中英文都能识别）
- [ ] `3_trading/trader.md` 的 `FINAL TRANSACTION PROPOSAL` 仍然正确折叠到 BUY/HOLD/SELL 三档（OVERWEIGHT→BUY、UNDERWEIGHT→SELL，已由优化 01 实现）

---

## 6. 测试计划

### 6.1 标的选择策略

挑选时优先满足"以前一直跑成 HOLD"的标的，便于验证新阈值能产出弱方向信号。建议候选：

- 协创数据（300857）—— 历史 9 次跑批中绝大多数为 HOLD
- 赣锋锂业（002460）—— 优化 01 验收时跑成 HOLD，可对比
- 任意一只港股 + 一只 ETF + 一只美股

### 6.2 必跑的关键对比

对每只标的：
1. 保留旧版报告（优化 01 验收时生成的 `20260509_2157xx` 系列）作为 baseline
2. 跑新版生成新时间戳报告
3. 重点对比：
   - **manager.md**：旧版评级是什么 → 新版评级是什么 → 敏感性自检结论是什么
   - **decision.md**：评级是否与 RM 一致；若不一致是否给出执行层理由
   - **trader.md**：FINAL 三档折叠是否正确（这一步本次不动，但要确认没回归）

### 6.3 评级分布健康度检查

跑完 5 只标的后，统计 RM 评级分布。健康的分布大致是：

```
BUY/SELL（强信号）：  20-40%
OVERWEIGHT/UNDERWEIGHT（弱信号）：  20-40%
HOLD：  20-40%
```

如果新版仍然是 100% HOLD 或 100% OVERWEIGHT，说明阈值调整未生效，需排查 prompt。

---

## 7. 风险与回滚

### 7.1 主要风险

- **LLM 把 OVERWEIGHT 当作"弱 BUY"用**：如果 prompt 里档位语义没讲清，LLM 输出会把 OVERWEIGHT 当成 BUY 的同义词，仓位建议依然激进。通过验收 5.1 的最后一项（OVERWEIGHT 仓位 ≤30%）检测
- **敏感性自检沦为形式**：LLM 可能写一段假大空的自检"假设论据弱 1 分，结论不变"。需在测试时人工抽查 2 份 manager.md，验证自检里的得分差计算是否真的有变化
- **PM 钻空子翻方向**：LLM 可能用执行层的借口反过来翻 RM 的方向。验收 5.2 的最后一项明确禁止跨方向翻转

### 7.2 回滚

- 所有改动应在**单个 commit** 中完成
- commit message 建议：`refactor(rating): 启用 5 档评级 + RM 敏感性自检 + PM 受控调整`
- 出现回归则 `git revert` 一键回退；返工时保持单 commit 原则

---

## 8. 不在本次范围

以下议题与本次改造相邻，但**本次不动**：

- **得分差计算的根本性改造**（结构化 rubric / 多维度独立打分）：本次只用敏感性自检兜底，不重做评分机制。如果敏感性自检效果有限，留待后续优化
- **signal_processing.py 简化为 5 档**：保持 7 级解析能力作为输入宽容兜底，不主动收窄
- **decision.md 显式列出与 trader 的分歧**（优化 04）：本次不做
- **风控三方辩论的 prompt**：不动
- **Trader 评级折叠规则**：优化 01 已写好，本次不动

---

## 9. 实施完成后

Qoder 完成后请回传：
1. 修改的文件列表（应只有 `research_manager.py` 和 `portfolio_manager.py`）
2. 跑过的 5 只标的的 ticker + 时间戳 + 最终评级
3. **任意 2 份 manager.md 的"敏感性自检"小节全文**（粘贴在回执里）
4. **如果有 PM 评级与 RM 不同的案例**，粘贴该 decision.md 的"评级调整说明"小节全文

由 Claude 复核验收标准后，更新 `docs/ROADMAP.md` 中本项状态为"已完成"，并开始撰写优化 03 的规格。
