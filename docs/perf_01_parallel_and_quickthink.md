# 性能优化 perf_01：4 分析师并行 + trader/bull/bear 切 quick_think

## 1. 背景与现状问题

### 1.1 端到端耗时基线

用户反馈一次股票分析跑批 **30-40 分钟**。基于代码结构估算的时间分布：

| 阶段 | LLM | 估时 | 是否串行 |
|------|-----|------|---------|
| 4 个分析师 + 工具循环 | deep_think | 12-20 min | **强制串行** |
| Bull → Bear (默认 1 round = 2 turns) | deep_think | 3-4 min | 串行（必然）|
| Research Manager | deep_think | 2-3 min | — |
| Trader | deep_think | 1.5-2 min | — |
| 3 风控辩手（默认 1 round = 3 turns）| quick_think | 3-4 min | 串行（必然）|
| Portfolio Manager | deep_think | 2-3 min | — |
| **合计** | | **24-38 min** | |

### 1.2 两个核心瓶颈

**瓶颈 A：4 分析师强制串行（最大）**

`tradingagents/graph/setup.py:158-184` 把 4 个分析师串成链：
```
START → Market → Social → News → Fundamentals → Bull Researcher
```

但**4 个分析师之间没有任何数据依赖**——每个只读自己的数据源，输出独立的 `*_report` 字段（market_report / sentiment_report / news_report / fundamentals_report）。下游 Bull/Bear researcher 才同时读 4 份。

**瓶颈 B：deep_think LLM 过度使用**

`tradingagents/graph/trading_graph.py:103-115` 把 trader、bull_researcher、bear_researcher 都固定走 deep_think_llm。但优化 01 把 trader 改造成**执行专家**后，它做的是结构化判断（仓位、止损、流动性算账），不需要深度推理。bull/bear researcher 是**修辞密度高、推理深度低**的任务（找论据反驳对方），quick_think 完全够用。

### 1.3 用户约束

- **MiniMax Starter 套餐**：并发上限未知（需用户查控制台），保守估计 5-20 并发
- **OOM 顾虑**：4 个分析师并行写入 `messages` list（add_messages reducer），峰值内存约 1-2MB，可控但需主动管理

---

## 2. 目标

完成后达到：

1. **端到端耗时降低 50%+**：从 30-40 min → 12-20 min
2. **保持输出质量基本一致**：4 份分析师报告内容不退化（因为彼此独立）；trader、bull/bear 切 quick_think 后输出质量在可接受范围
3. **MiniMax Starter 套餐可用**：默认 concurrency=4，但若触发 429 速率限制，提供清晰的 fallback 路径（一行配置改 concurrency=2）
4. **可回滚**：所有 LLM 模型选择都暴露 config flag，质量不达标时一行配置改回 deep_think

---

## 3. 改动范围

### 3.1 必须修改

| 文件 | 改动 |
|------|------|
| `tradingagents/default_config.py` | 新增 4 个配置项（concurrency + 3 个 LLM 选择 flag）|
| `tradingagents/graph/setup.py` | 重构图拓扑，支持 1/2/4 三种并发模式；fan-in 后插入 Merge Cleaner 节点 |
| `tradingagents/graph/trading_graph.py` | trader、bull_researcher、bear_researcher 默认 LLM 改 quick_think，但读 config flag 允许覆盖 |
| `tradingagents/agents/utils/agent_utils.py` 或新增 `tradingagents/agents/utils/merge_cleaner.py` | 新增 `create_merge_cleaner` 工具函数 |

### 3.2 不修改

- 任何 agent 的 prompt（market_analyst.py / news_analyst.py / social_media_analyst.py / fundamentals_analyst.py / bull_researcher.py / bear_researcher.py / trader.py / 各 manager / 各 risk debater）—— 不动
- `signal_processing.py`、`conditional_logic.py` —— 不动
- 4 个数据源 vendor / interface.py —— 不动
- minimax_client.py —— 不动（rate limit 由 ChatOpenAI 内置 retry 处理）

---

## 4. 详细变更

### 4.1 新增配置项

在 `tradingagents/default_config.py` 新增 4 项：

```python
# 性能优化配置
"analyst_concurrency": 4,           # 分析师并发上限。可选值: 1 (串行/原行为) / 2 (两波分组) / 4 (全并行)
"use_deep_for_trader": False,       # True 时 trader 用 deep_think_llm（rollback 用）
"use_deep_for_bull_researcher": False,   # True 时 bull researcher 用 deep_think_llm
"use_deep_for_bear_researcher": False,   # True 时 bear researcher 用 deep_think_llm
```

每项配置必须有注释说明默认值的选择理由。

### 4.2 P0：图拓扑重构（setup.py）

#### 4.2.1 设计：3 种并发模式

| concurrency | 图拓扑 | 说明 |
|-------------|--------|------|
| 1 | START → Market → Social → News → Fundamentals → Merge Cleaner → Bull | 当前串行行为，保留作为 fallback |
| 2 | START → [Market, Fundamentals] (并行) → barrier → [News, Sentiment] (并行) → Merge Cleaner → Bull | 两波分组，重 LLM (market/fundamentals) + 重工具调用 (news/sentiment) 配对 |
| 4 | START → [Market, Social, News, Fundamentals] (全部并行) → Merge Cleaner → Bull | 全并行，省时最大化 |

#### 4.2.2 实现要点

1. **保留每个分析师内部的 tool 循环**：`analyst → tools → analyst` 的循环结构不动，只改外层链接关系
2. **MsgClear 节点保留**：每个分析师后仍跟自己的 `Msg Clear xxx`，负责清理本分析师的工具调用消息
3. **Merge Cleaner 是 fan-in 汇聚点**：所有分析师分支的 `Msg Clear xxx` 都连接到同一个 `Merge Cleaner` 节点，再由它连到 `Bull Researcher`
4. **concurrency=2 的 barrier 实现**：把 4 个分析师拆成 2 组，第一组的 fan-in 节点连到第二组的 fan-out 起点。LangGraph 没有显式 barrier，用"中间汇聚节点"实现

#### 4.2.3 selected_analysts 兼容性

`setup_graph(selected_analysts)` 当前接受可选分析师列表。新逻辑必须保持兼容：
- 若 `selected_analysts` 长度 < `analyst_concurrency`，按实际数量并行
- 若 `selected_analysts = ["market"]`，串行行为（不需要 Merge Cleaner，但加上也无害）
- concurrency 取 `min(config["analyst_concurrency"], len(selected_analysts))`

### 4.3 P0：Merge Cleaner 节点

#### 4.3.1 职责

在 4 个分析师分支 fan-in 后、Bull Researcher 之前，做一次"messages 清扫"——确保 `state["messages"]` 不携带 4 路并行积累的工具调用残留进入下游。

#### 4.3.2 实现

参考现有 `create_msg_delete`（用于每个分析师后的 MsgClear）的实现模式。如果该函数已经返回 `{"messages": [RemoveMessage(id=...) for m in state["messages"]]}` 这种"全清"行为，可以复用；否则按相同模式新建。

实现位置建议：`tradingagents/agents/utils/merge_cleaner.py`（新文件）或追加到 `agent_utils.py`。

```python
def create_merge_cleaner():
    """Clear all accumulated messages from parallel analyst branches.
    
    Returns a node function that removes all current messages, ensuring
    downstream agents (Bull/Bear/RM/Trader/etc.) don't carry 4× worth of
    tool call residue from the parallel analyst phase.
    """
    def merge_cleaner_node(state):
        from langchain_core.messages import RemoveMessage
        return {"messages": [RemoveMessage(id=m.id) for m in state["messages"] if hasattr(m, "id") and m.id]}
    return merge_cleaner_node
```

注意：如果 `create_msg_delete` 已是这个模式，**直接复用它**作为 Merge Cleaner，不必造新函数。

### 4.4 P1：trader / bull / bear 切 quick_think

修改 `tradingagents/graph/trading_graph.py:103-115` 三处：

```python
# 旧代码（伪代码）
self.trader_llm = self._create_templllm(self.config.get("temperature_trader", 0.3), use_deep_think=use_deep)
self.bull_researcher_llm = self._create_templllm(self.config.get("temperature_bull_researcher", 0.5), use_deep_think=True)
self.bear_researcher_llm = self._create_templllm(self.config.get("temperature_bear_researcher", 0.5), use_deep_think=True)
```

改为：

```python
# 新代码（伪代码）
self.trader_llm = self._create_templllm(
    self.config.get("temperature_trader", 0.3),
    use_deep_think=self.config.get("use_deep_for_trader", False),  # 默认 quick_think
)
self.bull_researcher_llm = self._create_templllm(
    self.config.get("temperature_bull_researcher", 0.5),
    use_deep_think=self.config.get("use_deep_for_bull_researcher", False),  # 默认 quick_think
)
self.bear_researcher_llm = self._create_templllm(
    self.config.get("temperature_bear_researcher", 0.5),
    use_deep_think=self.config.get("use_deep_for_bear_researcher", False),  # 默认 quick_think
)
```

注意：**RM 和 PM 的 deep_think 不动**——他们做最终决策推理，需要保留。

### 4.5 不需要改的部分

- 4 个分析师本身的 prompt 和 LLM 选择（仍跟随 `use_deep_think_for_analysts`）
- 风控三方辩手（已是 quick_think）
- ConditionalLogic 的 should_continue 逻辑
- max_debate_rounds / max_risk_discuss_rounds 默认值
- 每个分析师的工具循环结构

---

## 5. 验收标准

### 5.1 性能指标（关键）

跑同一只标的 3 次取中位数（建议 002460 或 300857）：

- [ ] 端到端总耗时 ≤ **20 分钟**（baseline 30-40 min，目标降低 ≥40%）
- [ ] 4 分析师阶段耗时 ≤ **6 分钟**（baseline 12-20 min）
- [ ] PM 节点接收的 `state["messages"]` 长度 ≤ **5** 条（验证 Merge Cleaner 工作）
- [ ] 无 LLM 429 速率限制错误；若用户 MiniMax 配额低出现 429，spec 9.2 给出 fallback 路径

### 5.2 输出质量（防回归）

跑 6 只标的（沿用 03/04 标的：300857 / 002460 / 02616.HK / 159326 / AAPL / 300308）：

- [ ] 4 份分析师报告（market_report / sentiment_report / news_report / fundamentals_report）的**结构和篇幅**与 baseline 接近（±20% 篇幅，关键章节齐全）
- [ ] **trader.md 切换 quick_think 后**：仍包含 5 章节（方向确认/入场策略/止损位评估/流动性/执行风险），FINAL TRANSACTION PROPOSAL 折叠正确
- [ ] **bull/bear 辩论**切 quick_think 后：论据数量、引用具体数字的密度与 baseline 接近
- [ ] **最终评级**与 baseline 跑批比对：6 只标的中**至少 4 只**评级一致（允许 ±1 档差异，因为 LLM 切换天然有变化）
- [ ] 优化 01/02/02b/03 已通过的所有验收**无回归**：trader 不给方向、5 档评级、敏感性自检算术、PM 不跨方向翻转

### 5.3 系统层面

- [ ] CLI 完整跑通 6 只标的，覆盖 A 股 / 港股 / 美股 / ETF
- [ ] `signal_processing.SignalProcessor.process_signal` 评级解析正常
- [ ] `analyst_concurrency=1` 时退化为完全串行行为，与 baseline 完全一致（fallback 路径可用）
- [ ] `analyst_concurrency=2` 时按两波分组，最多 2 个 LLM 同时调用
- [ ] `use_deep_for_trader=True` 时 trader 切回 deep_think，与改造前行为一致

---

## 6. 测试计划

### 6.1 阶段 1：默认配置全量测试（concurrency=4 + 全 quick_think）

跑 6 只标的，记录每只的：
1. 端到端总耗时
2. 4 分析师阶段耗时
3. 最终评级
4. trader.md 章节完整性

### 6.2 阶段 2：fallback 路径验证

把 `analyst_concurrency` 改为 2，再跑 1 只标的，验证：
- 不报错
- 耗时介于 concurrency=1 和 concurrency=4 之间
- 输出质量一致

把 `analyst_concurrency` 改为 1，再跑 1 只标的，验证：
- 完全退化为旧串行行为
- 耗时与 baseline 一致

### 6.3 阶段 3：模型 rollback 验证

把 `use_deep_for_trader=True`，跑 1 只标的，验证 trader.md 用 deep_think 输出，质量与改造前一致。

### 6.4 关键对比

把 6 只标的的新版 trader.md / bull_history / bear_history 与 03 验收时的版本并排对比，记录：
- 内容深度变化（quick_think 是否明显变浅？）
- 结构是否保留
- 关键数字引用是否仍准确

---

## 7. 风险与回滚

### 7.1 主要风险

#### 风险 A：MiniMax Starter 套餐触发 429

**触发条件**：concurrency=4 时同时 4 路 LLM 长连接，可能超过 Starter 并发上限

**应对**：
1. 用户在 `default_config.py` 一行改 `analyst_concurrency: 2`，重跑无需其他改动
2. 若 concurrency=2 仍 429，改 1（退化串行，但仍享受 P1 的 quick_think 收益）
3. 提示用户在 config 里设置 `max_retries=3` 给 MiniMax 客户端（已支持透传）

#### 风险 B：quick_think 输出质量明显下降

**触发条件**：trader 或 bull/bear 切 quick_think 后，生成的内容明显变浅、关键数字引用错误、结构不完整

**应对**：3 个独立 config flag 可单独切回 deep_think，颗粒度细
- `use_deep_for_trader=True` 只切 trader 回去
- `use_deep_for_bull_researcher=True` 只切 bull 回去
- `use_deep_for_bear_researcher=True` 只切 bear 回去

**判断阈值**：验收 5.2 的"评级一致 ≥4/6"是硬指标。若 <4，必须切回 deep_think。

#### 风险 C：图拓扑改造引入死锁/无限循环

**触发条件**：fan-in 节点连接错误，或 Merge Cleaner 状态更新出问题

**应对**：先在 `analyst_concurrency=1` 模式下验证整体跑通（应该与 baseline 一致），再渐进开 2 / 4

#### 风险 D：Merge Cleaner 误删未来需要的消息

**触发条件**：Merge Cleaner 清空 messages，但下游某个 agent 实际依赖 messages（不只是 prompt 里读）

**应对**：
- 已确认 Bull/Bear/RM/Trader/PM 都从 state 字段（如 `*_report`、`investment_plan`）读输入，不读 messages
- 风控三方读 `report_context`（state 字段），不读 messages
- 如果验收时发现某 agent 异常，单独排查

### 7.2 回滚

- **完全回滚**：`git revert` 单 commit
- **部分回滚**（推荐）：通过 config flag 关掉子项目，避免重新部署
  - `analyst_concurrency: 1` → 关并行
  - `use_deep_for_trader: True` → 关 trader 模型降级
  - 等

### 7.3 commit 建议

`feat(perf): 4 分析师并行 + trader/bull/bear 切 quick_think (concurrency 默认 4)`

---

## 8. 不在本次范围

明确划出：

- **修改任何 agent 的 prompt/CoT**：本次纯结构和模型选择改造
- **修改 RM 或 PM 的 LLM**：保留 deep_think
- **修改风控三方辩手**：已是 quick_think
- **修改工具调用机制**：tools_market / tools_social / tools_news 的内部逻辑不动
- **修改数据源 / vendor**：不动
- **优化 04（决策卡）**：与本次正交并行推进，互不影响
- **客户端侧速率限制器**：不引入复杂的 semaphore/throttle，依赖 LangGraph 调度 + ChatOpenAI 内置 retry
- **缓存机制扩展**：checkpointer 不改

---

## 9. 实施完成后

Qoder 完成后请回传：

1. 修改的文件列表（应为 4 个文件：`default_config.py` / `setup.py` / `trading_graph.py` / `merge_cleaner.py` 或 `agent_utils.py`）

2. **性能数据**（必须提供，否则验收无法判断）：
   - baseline（concurrency=1 + 全 deep_think，跑 1 只标的）的端到端耗时
   - 默认配置（concurrency=4 + 全 quick_think，跑 1 只标的）的端到端耗时
   - 节省百分比

3. **6 只标的的最终评级表**（与 03 验收时的评级对比）：
   ```
   | 标的 | 03 baseline 评级 | perf_01 评级 | 是否一致 |
   ```

4. **任意 1 只标的的新版 trader.md 全文**（验证 quick_think 后质量）

5. **fallback 验证**：concurrency=2 和 concurrency=1 各跑 1 只标的的耗时和评级

由 Claude 复核：性能指标达标 + 评级一致性 ≥4/6 + 各 fallback 路径可用 → 更新 ROADMAP 中本项状态为 ✅，可继续推进 04（决策卡）或 05（入场时机）。
