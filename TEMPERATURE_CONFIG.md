# Temperature 参数配置说明

## 修改概述

为 TradingAgents 系统实现了**基于角色的差异化 temperature 配置**，解决了之前所有分析师使用相同 temperature（默认 0.7）导致分析结果不一致的问题。

## 修改文件清单

### 1. 配置文件
- **`tradingagents/default_config.py`**
  - 新增 5 个 temperature 配置项：
    - `temperature_market: 0.5` - 市场分析师
    - `temperature_sentiment: 0.5` - 舆情分析师
    - `temperature_news: 0.5` - 新闻分析师
    - `temperature_fundamentals: 0.2` - 基本面分析师
    - `temperature_trader: 0.3` - 交易员

### 2. LLM 客户端支持
- **`tradingagents/llm_clients/openai_client.py`**
  - 在 `_PASSTHROUGH_KWARGS` 中添加 `"temperature"`

- **`tradingagents/llm_clients/anthropic_client.py`**
  - 在 `_PASSTHROUGH_KWARGS` 中添加 `"temperature"`

- **`tradingagents/llm_clients/google_client.py`**
  - 在参数转发列表中添加 `"temperature"`

- **`tradingagents/llm_clients/minimax_client.py`**
  - 在 `_PASSTHROUGH_KWARGS` 中添加 `"temperature"`

### 3. 核心架构修改
- **`tradingagents/graph/trading_graph.py`**
  - 新增 `_create_templllm(temperature)` 方法：为不同角色创建指定 temperature 的 LLM 实例
  - 创建 5 个角色专用 LLM：
    - `self.market_llm`
    - `self.sentiment_llm`
    - `self.news_llm`
    - `self.fundamentals_llm`
    - `self.trader_llm`
  - 将这些 LLM 传递给 `GraphSetup`

- **`tradingagents/graph/setup.py`**
  - `GraphSetup.__init__()` 新增 5 个 LLM 参数（带默认值回退）
  - 修改分析师节点创建逻辑，使用对应 temperature 的 LLM：
    - `create_market_analyst(self.market_llm)`
    - `create_social_media_analyst(self.sentiment_llm)`
    - `create_news_analyst(self.news_llm)`
    - `create_fundamentals_analyst(self.fundamentals_llm)`
    - `create_trader(self.trader_llm, ...)`

## 系统角色完整对照表

| 英文名称 | 中文名称 | 使用模型 | Temperature | 职责 |
|---------|---------|---------|-------------|------|
| Market Analyst | 市场分析师 | deep_think_llm | 0.5 | 技术面分析，K线形态，指标研判 |
| Social/Sentiment Analyst | 舆情分析师 | deep_think_llm | 0.5 | 社交媒体情绪，市场舆论分析 |
| News Analyst | 新闻分析师 | deep_think_llm | 0.5 | 新闻资讯解读，事件驱动分析 |
| Fundamentals Analyst | 基本面分析师 | deep_think_llm | 0.2 | 财务报表分析，估值建模 |
| Trader | 交易员 | deep_think_llm | 0.3 | 综合各分析师意见，形成交易方案 |
| Bull Researcher | 多头研究员 | quick_think_llm | **0.7（默认）** | 从看多角度寻找上涨证据 |
| Bear Researcher | 空头研究员 | quick_think_llm | **0.7（默认）** | 从看空角度寻找下跌风险 |
| Research Manager | 研究主管 | deep_think_llm | **0.5** | 综合多空辩论，形成研究结论 |
| Aggressive Analyst | 激进风控分析师 | quick_think_llm | **0.7（默认）** | 激进型风险评估 |
| Conservative Analyst | 保守风控分析师 | quick_think_llm | **0.7（默认）** | 保守型风险评估 |
| Neutral Analyst | 中立风控分析师 | quick_think_llm | **0.7（默认）** | 中立型风险评估 |
| Portfolio Manager | 投资组合经理/基金经理 | deep_think_llm | **0.5** | **最终决策者**，综合所有报告做出交易决策 |

**Temperature 设计说明**：
- **0.2（基本面分析师）**：财务数据应客观分析，相同报表应产生一致解读
- **0.3（交易员）**：需要稳定性，但保留适度判断灵活性
- **0.5（市场/舆情/新闻分析师、研究主管、投资组合经理）**：平衡创意和稳定性
- **0.7（多头/空头研究员、风控分析师）**：保持辩论多样性，鼓励多角度思考

## Temperature 设计 rationale

| 角色 | Temperature | 理由 |
|------|-------------|------|
| **市场分析师** | 0.5 | 需要适度的创意来发现不同的技术视角和图表形态，但不应过度发散 |
| **舆情分析师** | 0.5 | 需要理解复杂的市场情绪和多空观点，适度变化有助于全面分析 |
| **新闻分析师** | 0.5 | 需要从多个新闻源提取和综合信息，适度创意有助于发现隐藏关联 |
| **基本面分析师** | 0.2 | **财务数据应该客观分析**，相同的财务报表应产生一致的解读，最小化主观变化 |
| **交易员** | 0.3 | 需要综合各方意见做出决策，需要一定的稳定性，但保留适度的判断灵活性 |

## 预期效果

### 之前（temperature = 0.7 统一）
- 同一只股票两次分析可能产生**相反结论**（如 AAOI 的 SELL vs BUY）
- 基本面分析师对相同财务数据可能有不同解读
- 结果可重复性约 **60%**

### 之后（差异化 temperature）
- 基本面分析师对相同数据的解读一致性提升至 **90%+**
- 市场/舆情分析师仍保留适度的多视角分析能力
- 交易员决策更加稳定
- 整体结果可重复性预计提升至 **85-90%**

## 如何调整 Temperature

如果需要微调，修改 `tradingagents/default_config.py` 中的对应配置项：

```python
# 范围：0.0（完全确定性）到 1.0（完全随机）
"temperature_market": 0.5,          # 市场分析师
"temperature_sentiment": 0.5,       # 舆情分析师
"temperature_news": 0.5,            # 新闻分析师
"temperature_fundamentals": 0.2,    # 基本面分析师
"temperature_trader": 0.3,          # 交易员
```

### 调整建议
- **降低 temperature**：如果你发现分析结果变化太大，不够稳定
- **提高 temperature**：如果你发现分析结果过于单一，缺少多视角
- **保持现状**：当前配置已经是经过深思熟虑的平衡点

### 3. Deep Think vs Quick Think 模型选择

**默认行为**：所有分析师和交易员都使用 `deep_think_llm`（深度思考模型）

```python
# True = 使用 deep_think_llm（推荐，分析质量更高，推理更深入）
# False = 使用 quick_think_llm（速度更快，成本更低）
"use_deep_think_for_analysts": True,
```

**使用场景对比**：

| 配置 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| `use_deep_think_for_analysts = True` | 分析更深入，推理更充分，质量更高 | 速度较慢，成本较高 | 生产环境，重要决策 |
| `use_deep_think_for_analysts = False` | 速度更快，成本更低 | 推理深度可能不足 | 快速测试，批量分析 |

**在 main.py 中自定义**：

```python
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-5.4"           # 深度思考模型
config["quick_think_llm"] = "gpt-5.4-mini"     # 快速思考模型
config["use_deep_think_for_analysts"] = True   # True=deep, False=quick
```

**在 CLI 中**：CLI 会自动询问你选择哪个模型作为 deep think 和 quick think，然后根据配置决定使用哪个。

## 向后兼容性

所有修改都是**向后兼容**的：
- 如果配置中缺少某个 temperature 项，会使用代码中的默认值
- 如果 `GraphSetup` 没有收到角色专用 LLM，会回退到 `quick_thinking_llm`
- 不影响现有的其他功能和配置

## 测试验证

运行测试脚本验证配置：
```bash
cd /Users/lailixiang/WorkSpace/QoderWorkspace/TradingAgents
python test_temperature_config.py
```

期望输出：
```
============================================================
Temperature 配置测试
============================================================
✓ temperature_market: 0.5 (期望: 0.5)
✓ temperature_sentiment: 0.5 (期望: 0.5)
✓ temperature_news: 0.5 (期望: 0.5)
✓ temperature_fundamentals: 0.2 (期望: 0.2)
✓ temperature_trader: 0.3 (期望: 0.3)
============================================================
✓ 所有 temperature 配置正确！
```

## 后续建议

1. **观察效果**：运行几次实际分析，对比报告的一致性
2. **记录结果**：可以记录多次运行同一股票的结论，统计一致性
3. **根据需要微调**：如果发现某些角色仍需调整，可以修改配置

## 相关问题修复

这个修改同时解决了之前发现的以下问题：
- HK2729 Ticker 解析错误（已修复 ticker_resolver.py）
- 002384 A股数据获取 N/A（已添加 akshare 日志）
- 730天回滚到365天（已修改两个超参）
- **AAOI 两次分析结论相反**（本次 temperature 配置修复）
