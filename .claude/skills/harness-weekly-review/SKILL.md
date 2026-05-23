---
name: harness-weekly-review
description: 跑 TradingAgents harness 周回测（backtest + review）+ 顺手扫 cron 健康度，把摘要通过飞书 webhook 推送。每周末用户手动触发一次。
---

# Harness Weekly Review

跑 TradingAgents 的周回测分析全流程，提取关键统计，通过飞书 webhook 一次性推送
"review 摘要 + cron 健康汇报"两件事。

## 何时使用

用户说以下任一关键词时调用此 skill：

- "跑下 harness 周 review"
- "harness 周末复盘"
- "weekly review"
- "/harness-weekly-review"
- "周末过一遍 harness 看下数据"

## 执行流程

所有命令都基于项目根目录 `/Users/lailixiang/WorkSpace/QoderWorkspace/TradingAgents`。

### 步骤 1：扫 cron 健康度（轻量，约 1 秒）

读最近 7 天的日志：

```bash
tail -300 harness_data/daily_update.log
```

按时间戳判定 3 类异常（任一为 true 就在飞书消息顶部加警示）：

- **漏跑严重**：7 天里运行次数 < 5
- **vendor 失败**：日志含字符串 `vendor 链全部失败（cache 已落后 N 天）`
- **程序异常**：日志含 `Traceback (most recent call last):`

正常场景：每天 21:00 附近 1 次运行，共 7 次，无 WARNING/Traceback。

### 步骤 2：跑回测切片统计

```bash
.venv/bin/python -m tradingagents.harness.backtest
```

预计 5-15 秒。无报错继续；报错时跳到"异常分支 A"。

### 步骤 3：生成 review 报告

```bash
.venv/bin/python -m tradingagents.harness.review
```

报告写到 `harness_data/reviews/backtest_<日期>.md`。无报错继续；报错时跳到"异常分支 A"。

### 步骤 4：找最新的 review 报告

```bash
ls -t harness_data/reviews/backtest_*.md | head -1
```

### 步骤 5：从报告里提取关键统计

读文件后用 LLM 语义提取（不要写脆弱的 regex）：

- **总样本数**：从"全局命中率（所有样本）"表里加总 4 个 horizon 的样本数
- **4 个 horizon 命中率**：T / T+1 / T+5 / T+30 各自一个百分比
- **偏差告警条数**：扫"系统性偏差（按严重度排序）"段，统计 ⚠️ high 和 🔶 medium 两类条数
- **Top 3 偏差摘要**：取按严重度排序的前 3 条，每条 ≤40 字摘要

如果总样本数 < 5，跳到"异常分支 B：样本不足"。

### 步骤 6：拼装飞书消息正文

固定格式（多行 text，注意换行符在 shell 里要正确转义）：

```
【TradingAgents 周末 Review】<日期>

cron 健康度：<✓ 正常 或 ⚠ 详细异常说明>

回测核心统计：
- 总样本数：<X>
- T 命中率：<X>%
- T+1 命中率：<X>%
- T+5 命中率：<X>%
- T+30 命中率：<X>%
- 偏差告警：高严重度 <N> 条 / 中严重度 <M> 条

Top 3 偏差：
1. <第一条摘要>
2. <第二条摘要>
3. <第三条摘要>

报告位置：harness_data/reviews/backtest_<日期>.md
下一步：用 Claude 详细阅读完整报告，决定本周改哪些 prompt/阈值
```

### 步骤 7：发飞书

调本 skill 内的 helper 脚本：

```bash
./.claude/skills/harness-weekly-review/send_feishu.sh "<消息正文>"
```

helper 内部封装 curl + JSON 转义。webhook URL 已写死在脚本里。

### 步骤 8：终端打印执行摘要

告诉用户：
- 报告位置（绝对路径）
- 飞书发送状态（成功/失败 + 飞书返回码）
- 是否有 cron 异常需要单独关注

## 异常分支

### 分支 A：backtest 或 review 命令报错

不要继续后续步骤。直接发飞书报错消息：

```
✗ TradingAgents 周末 Review 执行失败

报错命令：<backtest 或 review>
错误摘要：<stderr 第一行或关键错误>

请手动跑一次确认问题：
cd /Users/lailixiang/WorkSpace/QoderWorkspace/TradingAgents
.venv/bin/python -m tradingagents.harness.backtest
```

### 分支 B：样本量不足（总样本 < 5）

跳过 Top 3 偏差提取（没意义），发简短消息：

```
⚠ TradingAgents 周末 Review

本周样本量仅 <X> 条（< 5），统计意义弱。

cron 健康度：<✓ 或 异常说明>

建议：再积累 2-3 周数据后再做正式 review。
报告位置：harness_data/reviews/backtest_<日期>.md
```

### 分支 C：飞书 webhook 发送失败

`send_feishu.sh` 返回非零退出码时，**不要静默吞错**。
在终端明确告诉用户：
- curl 返回的具体错误（飞书 code≠0 / 网络错 / etc.）
- 消息正文（让用户自己手动粘贴到飞书或者查问题）

## 设计原则

- **机械动作放 shell**：curl / json 转义 / chmod 都在 `send_feishu.sh` 里
- **语义判断放 LLM**：解读日志异常类型、解读 review 报告内容、拼消息正文
- **失败不静默**：每一步报错都告诉用户，不要让用户以为系统正常但实际没工作

## 调度方式（V1）

每周末手动触发一次。用户在 Claude 对话里说关键词即可。

未来如需自动化，可以用 macOS cron 配合 Claude Code 的 batch 模式（`claude -p ...`），但 V1 不做。
