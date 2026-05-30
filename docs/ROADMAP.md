# TradingAgents 优化路线图

## 顶层评审

完整的框架评审（11 个 agent 的人设/CoT/I/O 诊断 + 链路级问题）见 [FRAMEWORK_REVIEW.md](FRAMEWORK_REVIEW.md)。本路线图是评审结论的实施排期。

---

## 协作模式

- **架构师（Claude）**：分析现状、撰写优化规格文档、定义验收标准
- **实施者（Qoder，另一个 AI coding agent）**：根据规格文档修改代码
- **项目主（用户）**：在两者之间传递信息、决定优先级、最终验收

每一项优化对应一份独立的规格文档（`NN_xxx.md`）。Qoder 拿到一份文档后即可独立完成实现，无需查看本路线图或其他文档。

---

## 优化项清单

按依赖关系和价值排序，**逐项推进**（不并行）。每完成一项并验收后，再开始下一项。

| 序号 | 文档 | 主题 | 状态 |
|------|------|------|------|
| 01 | [01_trader_role_redesign.md](01_trader_role_redesign.md) | Trader 角色从"二次决策者"改为"执行专家" | ✅ 已完成（2026-05-09 验收通过，[验收报告](01_trader_role_redesign_验收报告.md)） |
| 02 | [02_rating_threshold_relaxation.md](02_rating_threshold_relaxation.md) | 放松 HOLD 阈值，启用 OVERWEIGHT/UNDERWEIGHT | ✅ 已完成（含 02b 补丁，2026-05-10 验收通过） |
| 02b | [02b_sensitivity_check_patch.md](02b_sensitivity_check_patch.md) | 修复敏感性自检的边界包含 + 压力测试方法论 | ✅ 已完成（2026-05-10 验收通过） |
| 03 | [03_sensitivity_arithmetic_unification.md](03_sensitivity_arithmetic_unification.md) | 敏感性自检算术方法论统一（避免不同标的 Δd 差 6 倍） | ✅ 已完成（2026-05-10 验收通过：5 档评级全产出 BUY/OVERWEIGHT/HOLD/UNDERWEIGHT/SELL 各 1，AAPL Δd 异常彻底消除）|
| 04 | [04_decision_card.md](04_decision_card.md) | **决策卡（Decision Card）标准化**——decision.md 头部 9 字段摘要表 | 🟡 条件通过（结构层面 100% 过，[中期状态](04_decision_card_中期状态.md)；待 04b 补丁修复 3 处字段硬伤）|
| 04b | [04b_decision_card_patch.md](04b_decision_card_patch.md) | 决策卡补丁——置信度映射 + 入场时机 + 催化/风险格式 | 待实施 |
| 05 | 05_entry_timing_and_weights.md | **入场时机决策模块 + 分析师权重显式化**——评级与时机二维独立 | P0，待规格 |
| 06 | 06_fundamentals_dupont_peers.md | fundamentals_analyst 引入 DuPont 分解 + 同业可比公司估值 | P1，待规格 |
| 07 | 07_market_regime_first.md | market_analyst 前置市场制度（regime）判断，按 regime 选指标 | P1，待规格 |
| 08 | 08_risk_committee_redesign.md | 风控辩论职能化重构（激进/保守/中立 → 宏观/头寸/尾部）| P2，待规格 |
| 09 | 09_thesis_template.md | bull/bear researcher 强制 thesis 模板 + 论据强度自评 | P2，待规格 |

**原优化 03（"PE 数据 sanity check 上移"）的处置**：发现 `agent_utils.validate_fundamentals_data` 已实现 PE×EPS、PE>500、动/静 PE 比率等 4 项确定性校验，且在 fundamentals_analyst 已被调用。原计划范围已被前人实现，剩余工作只是清理下游冗余 prompt → 降级为优化 05。

**原优化 04（"列出与 trader 分歧"）的处置**：优化 01 改造 trader 后，"分歧"语义已变（trader 不再给方向）。该议题合并进新优化 05 的"入场时机"模块（PM 对 trader 执行风险的态度本身就是分歧承载点）。

**P3 后续候选**（不入主路线，回流候选）：
- ETF 流动性数据注入 trader（优化 01 副产物）
- 同标的历史决策对照（PM 输入注入 last decision summary）
- 反思机制扩展到无 returns_losses 场景（无回灌时也能学习）
- sentiment_analyst 重定位为"资金面分析师"（北向/龙虎榜/融资余额取代 Xueqiu）

### 性能优化专项（与主路线 03-09 并行推进）

主路线 03-09 是 prompt/CoT/输出格式层面的改造。性能专项专注**降低端到端跑批耗时**（当前 30-40 min/标的），与 prompt 优化解耦。

| 序号 | 文档 | 主题 | 状态 |
|------|------|------|------|
| perf_01 | [perf_01_parallel_and_quickthink.md](perf_01_parallel_and_quickthink.md) | 4 分析师并行（concurrency=4）+ trader/bull/bear 切 quick_think | 🟠 已回退（[验收报告](perf_01_验收报告.md) 暴露测试条件不可比 + 评级 2/3 下移；P0 并行部分回退，保留 P1 quick_think flags）|
| perf_02 | [perf_02_rollback_to_quickthink_only.md](perf_02_rollback_to_quickthink_only.md) | 回退 perf_01 并行图，仅保留 P1（quick_think 模型选择 config flag）| 待实施 |

性能专项与主路线的关系：
- 不依赖：性能改造不依赖主路线任何完成项
- 不冲突：性能改造只动 `setup.py` 图拓扑和 `trading_graph.py` LLM 选择，不动任何 agent 的 prompt
- 可并行：可在 04 实施前/中/后任何时点插入推进

### 已完成项的观察清单（不计入本项验收，留作后续）

- **优化 01 副产物**：159326 ETF 的 trader.md "流动性"章节出现"需查询近期数据"占位文本，说明 ETF 类标的的日均成交额未被注入 trader context。建议在后续优化中考虑把流动性元数据作为 trader 的额外输入（与本路线图的 4 项均不冲突，可作为优化 05 候选）。
- **优化 02 副产物**：原 spec 的硬验收项"≥2 只 OW/UW"在敏感性自检存在的前提下结构性不可达——OW/UW 区间宽度仅 0.5，且自检会主动把脆弱的 OW/UW 降到 HOLD。300857 中间步骤产出 UNDERWEIGHT 后被正确降档已证明机制可用。已不再追求"必须产出 OW/UW"作为硬指标。
- **优化 02b 遗留观察项**（不阻塞，未来视实际行为决定是否处理）：
  - **算术方法论未统一**：弱化 2 条论据 1 分对最终 d 的影响幅度在不同标的上差 6 倍（02616.HK Δd≈0.33 vs AAPL Δd≈2.0），原因是 LLM 在"sum 还是 avg"上没收敛。这影响敏感性自检在边界 case 的命中率
  - **多档跨越仅降 1 档**：AAPL 自检 d 从 -0.7 跳到 +1.3（跨 3 档），按现规则只降 1 档到 HOLD。若未来出现 SELL→OVERWEIGHT 这种跨方向跳，单档降只能到 UNDERWEIGHT，不够。可考虑"跨方向直接砸 HOLD"兜底
  - **可能存在系统性看空偏倚**：02b 验收 5/5 标的最终全部 SELL/HOLD。可能是市场情绪+小样本，也可能是 prompt 改造后副作用。未来跑批多了再回看
- **优化 04 的前置变化**：trader 已经不再给方向，所以"显式列出与 trader 的分歧"在原始定义下不再适用。优化 04 的规格在撰写时需要重新定义"分歧"——可能改为"列出 trader 标记的执行风险中，portfolio_manager 选择忽略 vs 采纳的部分"。

---

## 依赖与排序理由

```
01 (trader 重构) ─┬─→ 04 (分歧记录的定义取决于 trader 是否给方向)
                  └─→ （影响 portfolio_manager prompt 措辞）

02 (评级阈值) ────→ 影响所有报告的评级分布，应在大批回归测试前完成

03 (数据校验) ────→ 与上述独立，可单独推进
```

实际推进顺序：**01 → 02 → 03 → 04**

---

## 规格文档统一结构

每份 `NN_xxx.md` 都包含以下章节，便于 Qoder 解析：

1. **背景与现状问题**：为什么要做这件事，引用具体报告/代码作为证据
2. **目标**：完成后的预期状态（一句话能讲清楚）
3. **改动范围**：涉及哪些文件，明确不改哪些文件
4. **详细变更**：按文件列出该改什么、保留什么
5. **验收标准**：可客观验证的清单
6. **测试计划**：跑哪几只标的、对比哪些输出
7. **风险与回滚**：可能的副作用、出问题怎么撤
8. **不在本次范围**：明确划出本次不动的相邻议题

---

## 验收哲学

- **跑真实报告**：每项验收都需要在 3-5 只标的（覆盖 A 股 / 港股 / 美股 / ETF 至少 2 类）上跑完整 pipeline，对比新旧 `reports/<ticker>/<timestamp>/` 输出
- **可客观验证**：验收标准必须能用 grep / 文本对比验证，不能只靠"读起来更好"
- **不允许回归**：评级解析（`signal_processing.py`）必须仍能从最终输出中提取出有效评级，CLI 跑通无报错

---

## 文档迭代规则

- Qoder 完成实施后，由用户回传结果给 Claude
- Claude 复核：达标则在本路线图把状态改为"已完成"，并开始写下一份规格
- 不达标则在原规格文档底部追加"返工说明"小节，Qoder 据此二次修改
