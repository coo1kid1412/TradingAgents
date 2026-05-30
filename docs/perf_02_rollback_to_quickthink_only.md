# 性能优化 perf_02：回退并行图，仅保留 P1（quick_think 模型选择）

## 1. 背景

perf_01 验收暴露三个不可调和的问题：
- 性能数据无法证明 spec 目标达成（测试环境 deep=quick 同模型、辩论轮数 1 vs baseline 2、单只标的样本无中位数）
- 2/3 标的评级下移 1 档，无法分辨是 LLM 噪声还是并行改造的副作用
- 6 只标的完整测试 + 港股/美股/ETF 覆盖 + PM messages 长度验证 + use_deep_for_trader rollback 均未完成

用户决定：**回退并行图拓扑，仅保留 P1（trader / bull / bear 走 quick_think 的 config flag）**。

P0 并行的设计和实现工作不丢——`docs/perf_01_parallel_and_quickthink.md` 和 `docs/perf_01_验收报告.md` 保留作为未来重启该工作的参考。

---

## 2. 目标

- 图拓扑回到 perf_01 之前的纯串行版本（4 分析师 → Bull → ...）
- 保留 P1 的 3 个 LLM 选择 config flag（默认 quick_think，可单独 rollback）
- 删除并行相关代码与配置，避免遗留死代码与配置噪音

---

## 3. 改动范围

### 3.1 必须修改

| 文件 | 改动 |
|------|------|
| `tradingagents/graph/setup.py` | 完全回退到 perf_01 之前的串行版本 |
| `tradingagents/default_config.py` | 删除 `analyst_concurrency` 配置项 |
| `tradingagents/agents/utils/agent_utils.py` | 删除 `create_merge_cleaner` 函数及其引入的 `RemoveMessage`/`HumanMessage` 导入（如这些导入未被其他函数使用）|
| `tradingagents/graph/trading_graph.py` | `setup_graph` 调用处去掉第二个参数 `self.config`（第 173 行）|

### 3.2 必须保留（不动）

| 文件 | 保留内容 |
|------|---------|
| `tradingagents/default_config.py` | 保留 `use_deep_for_trader` / `use_deep_for_bull_researcher` / `use_deep_for_bear_researcher` 三个配置项（默认 False = quick_think）|
| `tradingagents/graph/trading_graph.py` | 保留第 109-116 行的 LLM 选择改造（trader/bull/bear 读取 config flag）|

---

## 4. 详细变更

### 4.1 setup.py 回退（最大改动）

回退到 perf_01 之前的版本：
- 删除 `_assign_analyst_waves` 函数
- 删除 `_create_barrier_node` 函数
- 删除 `_create_done_node` 函数
- 删除 `Merge Cleaner` 节点的创建和接入
- 删除 `is_parallel` 双模式分支（exit_nodes 字典）
- 删除 wave-based 边连接逻辑
- 删除 `from tradingagents.agents.utils.agent_utils import create_merge_cleaner` 导入

恢复为：
- 4 个分析师顺序连接（market → social → news → fundamentals → Bull Researcher）
- 每个分析师的 exit 节点固定用 `create_msg_delete()`
- `setup_graph(self, selected_analysts=[...])` 签名去掉 `config` 参数

**参考**：原始串行版本可以从 git 历史中找回（用 `git show HEAD:tradingagents/graph/setup.py` 因为 perf_01 改动尚未提交，HEAD 仍是原始版本）。直接 `git checkout HEAD -- tradingagents/graph/setup.py` 是最干净的回退方式。

### 4.2 default_config.py 删除一行

删除（第 48 行附近）：
```python
"analyst_concurrency": 4,               # 分析师并发上限。1=串行(原行为) / 2=两波分组(...) / 4=全并行
```

保留以下三行（第 49-51 行）：
```python
"use_deep_for_trader": False,
"use_deep_for_bull_researcher": False,
"use_deep_for_bear_researcher": False,
```

### 4.3 agent_utils.py 删除 create_merge_cleaner

删除 `create_merge_cleaner` 整个函数（约第 70-84 行）及其依赖的 `RemoveMessage` / `HumanMessage` 导入——**前提是这些导入未被其他函数使用**。请用 grep 确认：

```bash
grep -n "RemoveMessage\|HumanMessage" tradingagents/agents/utils/agent_utils.py
```

如果只有 `create_merge_cleaner` 内部用，连导入一起删；如果其他地方还在用，只删函数体保留导入。

### 4.4 trading_graph.py 调用点修正

第 173 行：
```python
self.graph = self.graph_setup.setup_graph(selected_analysts, self.config)
```

改回：
```python
self.graph = self.graph_setup.setup_graph(selected_analysts)
```

---

## 5. 验收标准

### 5.1 系统层面

- [ ] CLI 完整跑通 1 只标的（建议 300308，便于和 04 baseline 比对），无报错
- [ ] 跑批耗时回到 perf_01 之前的基线水平（~30-35 min）——这是预期，不是回归
- [ ] 评级与 04 验收时同一标的的评级一致（HOLD，因为 300308 串行模式下评级稳定）

### 5.2 P1 flag 验证

- [ ] 默认配置（`use_deep_for_trader=False`）下跑通一次，确认 trader 走 quick_think
- [ ] 把 `use_deep_for_trader=True`，重跑 1 只标的，确认 trader 走 deep_think 时仍正常出 trader.md

### 5.3 死代码清理

- [ ] `grep -rn "analyst_concurrency\|create_merge_cleaner\|_assign_analyst_waves\|_create_barrier_node\|_create_done_node" tradingagents/` 应**无输出**（除 docs/perf_01_*.md 历史档案）
- [ ] `grep -rn "Merge Cleaner" tradingagents/` 应**无输出**

### 5.4 无回归

- [ ] 优化 01/02/02b/03/04 已通过的所有验收无回归（4 个 manager.md / trader.md / decision.md 章节齐全、5 档评级、敏感性自检、决策卡）

---

## 6. 测试计划

跑 1 只标的（300308）两次：
1. 默认配置（quick_think 全开）：耗时 + 评级
2. `use_deep_for_trader=True`：耗时 + trader.md 是否走 deep

不需要跑 6 只——本次只是回退，不是新功能。

---

## 7. 风险与回滚

### 7.1 风险

- **遗漏死代码**：忘记清理某个 import 或某段并行边连接代码，导致 Python 启动错误。验收 5.3 的 grep 检查覆盖这点
- **保留的 P1 改动出问题**：如 `use_deep_for_trader` flag 读取失败、temperature 配置错位。验收 5.2 验证

### 7.2 回滚

如果回退本身出错，最简方案：
```bash
git checkout HEAD -- tradingagents/graph/setup.py tradingagents/default_config.py tradingagents/agents/utils/agent_utils.py tradingagents/graph/trading_graph.py
```
（前提是 perf_01 改动尚未 commit）

然后从零重做本 spec。

---

## 8. 不在本次范围

- 重新评估 P0 并行的设计（perf_01 spec 已存档，未来若想重启从那份开始）
- 修改 quick_think_llm 配置（当前 `MiniMax-M2.7` 同时是 deep_think_llm，P1 无实际加速效果——这是用户层面的配置决策，不在本次代码改动范围）
- 任何 agent 的 prompt 改动

---

## 9. 实施完成后

Qoder 完成后请回传：
1. 修改的文件列表
2. `grep` 死代码检查的输出（应为空）
3. 300308 默认配置跑批的最终评级 + 耗时
4. `use_deep_for_trader=True` 跑批的 trader.md 中"方向确认"小节首句（确认走 deep_think 仍正常）

由 Claude 复核后更新 `docs/ROADMAP.md`，将 perf_01 状态改为"已回退（保留 P1 quick_think flags）"，并将下一步重心切回主路线 04（决策卡）。
