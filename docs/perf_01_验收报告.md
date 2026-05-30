# perf_01 性能优化验收报告

> 日期：2026-05-11
> Spec：`docs/perf_01_parallel_and_quickthink.md`

---

## 1. 实现概要

### 1.1 修改文件

| 文件 | 修改内容 |
|------|---------|
| `tradingagents/default_config.py` | 新增 4 个配置项：`analyst_concurrency`, `use_deep_for_trader`, `use_deep_for_bull_researcher`, `use_deep_for_bear_researcher` |
| `tradingagents/agents/utils/agent_utils.py` | 新增 `create_merge_cleaner()` 函数，fan-in 后清理所有并行分支的残留消息 |
| `tradingagents/graph/trading_graph.py` | trader/bull/bear 的 LLM 选择改为读取 config flag（默认 quick_think） |
| `tradingagents/graph/setup.py` | 完全重构图拓扑，支持 concurrency=1/2/4 三种模式 |

### 1.2 图拓扑设计

| 模式 | 拓扑 | 说明 |
|------|------|------|
| `concurrency=4` | START → 4 analysts (parallel) → Merge Cleaner → Bull | 全并行，单波次 |
| `concurrency=2` | START → wave1(2 analysts) → wave2(2 analysts) → Merge Cleaner → Bull | 两波分组，波次间通过 exit→analyst 直接边同步（无 barrier 节点） |
| `concurrency=1` | START → analyst1 → analyst2 → analyst3 → analyst4 → Merge Cleaner → Bull | 纯串行链，完全退化旧行为 |

### 1.3 关键设计决策

1. **Msg Clear 节点双模式**：并行模式 (`is_parallel=True`) 下，Msg Clear 节点使用 `_create_done_node()`（pass-through，返回 `{}`），避免并行分支的 `RemoveMessage` 冲突；串行模式使用原始 `create_msg_delete()`
2. **Merge Cleaner**：并行模式专属，fan-in 后清理所有消息并注入 "Continue" placeholder，确保 PM 节点接收干净的 state
3. **消除 barrier 节点**：初版使用 `_create_barrier_node()` 做 wave 同步，但 LangGraph 内部分支追踪导致下游 `risk_debate_state` 的 `LastValue` channel 收到多值冲突。修复方案：改为当前波次 exit 节点直接指向下一波次 analyst 节点（fan-in 同步），无中间节点

---

## 2. 耗时对比

### 2.1 测试条件

- LLM: MiniMax-M2.7（deep_think_llm = quick_think_llm = 同一模型）
- `_DEBATE_ROUNDS = 1`（测试配置）
- 数据源: yfinance + Tushare + AKShare

### 2.2 测试结果

| 模式 | 标的 | 开始时间 | 结束时间 | 耗时 | 评级 |
|------|------|---------|---------|------|------|
| baseline 串行 (04 验收) | 300308 | ~23:50 | 00:25 | **~35 min** | HOLD |
| concurrency=4 | 300857 | ~00:50 | 01:17 | **~27 min** | SELL |
| concurrency=2 | 002460 | 01:54 | 02:14 | **~20 min** | UNDERWEIGHT |
| concurrency=1 | 300308 | ~02:09 | 02:39 | **~30 min** | HOLD |

### 2.3 耗时分析

- **concurrency=4 vs baseline**：27 min vs ~35 min → **节省 ~23%**。P0 并行收益被以下因素抵消：(1) 当前 deep_think_llm = quick_think_llm，P1 模型降级无实际效果；(2) 下游 debate/trader/risk/PM 仍为串行
- **concurrency=2**：20 min，因标的不同（002460 数据量较小）导致绝对值偏低，不宜直接横向对比
- **concurrency=1**：30 min，与 baseline 的 35 min 接近，证实串行退化正常

> **注意**：不同标的的数据获取时间和 LLM 推理时间差异较大，以上耗时仅供参考。严格的对比需要同一标的多次跑取中位数。

---

## 3. 评级对照表（vs 03/04 基线）

| 标的 | 03/04 基线评级 | perf_01 评级 | 并发模式 | 差异档位 | 说明 |
|------|-------------|------------|---------|---------|------|
| 300857 | UNDERWEIGHT | SELL | c=4 | -1档 | 均为看空方向，SELL 比 UNDERWEIGHT 更悲观 |
| 002460 | HOLD | UNDERWEIGHT | c=2 | -1档 | 均为偏空，UNDERWEIGHT 比 HOLD 更悲观 |
| 300308 | HOLD | HOLD | c=1 | 一致 | 串行模式，结果应与基线一致 |

**分析**：3 只标的中 1 只评级完全一致，2 只有 ±1 档差异。差异原因：
1. **测试使用 `_DEBATE_ROUNDS=1`**（基线为 2），辩论轮数减少导致 RM 得分分布不同
2. **LLM 非确定性**：即使同一配置，MiniMax-M2.7 的输出也存在波动
3. **数据时效性**：两次分析间隔数小时，市场数据可能略有变化

> spec 5.2 要求"6 只标的中至少 4 只评级一致（允许 ±1 档差异）"。当前仅 3 只数据，2 只在 ±1 档内。需补充完整 6 标的测试（concurrency=4 + `_DEBATE_ROUNDS=2`）才能最终判定。

---

## 4. Fallback 验证

### 4.1 concurrency=2

| 检查项 | 结果 |
|--------|------|
| 不报错 | ✅ 无 `InvalidUpdateError` 或其他异常 |
| 正常出报告 | ✅ 报告保存在 `reports/002460_赣锋锂业_20260511_021403` |
| 4 份分析师报告齐全 | ✅ market/sentiment/news/fundamentals 均有输出 |
| 最终评级输出 | ✅ UNDERWEIGHT |

### 4.2 concurrency=1

| 检查项 | 结果 |
|--------|------|
| 完全退化为旧串行行为 | ✅ 图拓扑为 analyst1→analyst2→...→Merge Cleaner 链式 |
| 不报错 | ✅ 无任何错误 |
| 正常出报告 | ✅ 报告保存在 `reports/300308_中际旭创_20260511_023939` |
| 耗时与 baseline 接近 | ✅ 30 min vs baseline ~35 min |

---

## 5. 已知问题与修复记录

### 5.1 RemoveMessage 并行冲突（已修复）

**现象**：concurrency=4 初版测试报错 `"invalid params, tool result's tool id not found"`

**根因**：4 个分析师并行运行时，每个分支的 `create_msg_delete()` 产生的 `RemoveMessage` 操作会删除其他分支的 tool_call ID，导致 fan-in 合并状态时 API 报 400 错误

**修复**：并行模式下 Msg Clear 节点替换为 `_create_done_node()`（pass-through），所有消息清理延迟到 Merge Cleaner 统一处理

### 5.2 Barrier 节点导致 risk_debate_state 冲突（已修复）

**现象**：concurrency=2 测试报错 `At key 'risk_debate_state': Can receive only one value per step`

**根因**：使用独立 `_create_barrier_node()` 做 wave 同步时，LangGraph 内部对 fan-in/fan-out 的分支追踪导致下游风控循环节点被视为同一 step 的并行节点，`risk_debate_state`（LastValue channel）收到多值

**修复**：消除 barrier 节点，改为当前波次 exit 节点直接指向下一波次 analyst 节点（形成 fan-in 同步），无中间节点。修改后 concurrency=2 测试通过

---

## 6. 验收标准逐项检查

### 6.1 性能指标（spec 5.1）

| 标准 | 结果 | 状态 |
|------|------|------|
| 端到端 ≤20 min | concurrency=2 达到 20 min，c=4 为 27 min | ⚠️ 部分达到 |
| 4 分析师阶段 ≤6 min | 未单独计时（日志未输出阶段耗时） | ⚠️ 待补充 |
| PM 节点 messages ≤5 条 | 未检查 | ⚠️ 待补充 |
| 无 429 限流 | c=4/c=2/c=1 均无 429 | ✅ |

### 6.2 输出质量（spec 5.2）

| 标准 | 结果 | 状态 |
|------|------|------|
| 4 份分析师报告结构与基线接近 | 目视确认，篇幅和章节完整 | ✅ |
| trader.md 5 章节完整 | 有方向确认/入场策略/止损位/流动性/执行风险 | ✅ |
| 最终评级与基线 ±1 档 | 3 只中 1 只一致，2 只 ±1 档 | ⚠️ 需 6 只完整数据 |
| 无回归（01/02/03 验收项） | 未做专项回归测试 | ⚠️ 待补充 |

### 6.3 系统层面（spec 5.3）

| 标准 | 结果 | 状态 |
|------|------|------|
| CLI 跑通 A 股/港股/美股/ETF | 仅测试 A 股（300857/002460/300308） | ⚠️ 待补充 |
| analyst_concurrency=1 退化为串行 | ✅ 300308 串行测试通过 | ✅ |
| analyst_concurrency=2 两波分组 | ✅ 002460 测试通过 | ✅ |
| use_deep_for_trader=True rollback | 未单独测试 | ⚠️ 待补充 |

---

## 7. 待完成事项

- [ ] **完整 6 标的测试**（concurrency=4, `_DEBATE_ROUNDS=2`）：300857 / 002460 / 02616.HK / 159326 / AAPL / 300308
- [ ] **补充阶段耗时**：在日志中增加 4 分析师阶段的起止时间戳
- [ ] **PM messages 长度验证**：确认 Merge Cleaner 工作后 PM 接收的消息数 ≤5
- [ ] **港股/美股/ETF 覆盖**：验证 02616.HK / AAPL / 159326 在并行模式下正常
- [ ] **模型 rollback 测试**：`use_deep_for_trader=True` 验证 trader 切回 deep_think

---

## 8. 报告路径

| 测试 | 报告路径 |
|------|---------|
| concurrency=4 (300857) | `reports/300857_协创数据_20260511_011733` |
| concurrency=2 (002460) | `reports/002460_赣锋锂业_20260511_021403` |
| concurrency=1 (300308) | `reports/300308_中际旭创_20260511_023939` |
