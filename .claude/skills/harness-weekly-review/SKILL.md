---
name: harness-weekly-review
description: 触发 TradingAgents harness 周回测——跑 backtest+review、扫 cron 健康度、推送摘要到飞书。每周六上午 10:00 由 cron 自动跑；用户也可手动触发或要求 Claude 做深度分析。
---

# Harness Weekly Review

## 两种触发场景

### 场景 A：cron 自动触发（主路径）

每周六上午 10:00 由系统 cron 调 `weekly_review.py`，全流程自动完成，飞书会收到结构化摘要。
**用户不需要做任何事**，被动收消息即可。

### 场景 B：Claude 中手动触发（深度分析路径）

用户在 Claude 对话里说以下关键词时调用此 skill：

- "跑下 harness 周 review"
- "weekly review"
- "/harness-weekly-review"
- "深度分析下这周回测"

## 执行流程

### 第一阶段：跑数据（machine 部分，调 Python 脚本）

```bash
cd /Users/lailixiang/WorkSpace/QoderWorkspace/TradingAgents
.venv/bin/python -m tradingagents.harness.weekly_review
```

这一步会：
1. 扫 daily_update.log 检查 cron 健康度
2. 跑 backtest（重算切片）
3. 跑 review（生成 markdown）
4. 拼装摘要并推送飞书（结构化、固定格式）
5. 终端打印 `[N/6]` 进度

脚本退出码 0 = 成功（飞书收到）；非 0 = 失败（终端有错误信息）。

### 第二阶段：深度解读（LLM 部分，仅场景 B 触发）

cron 触发时第一阶段就结束。但用户手动触发时，往下做以下额外分析：

1. 读最新的 review 报告：
   `cat $(ls -t harness_data/reviews/backtest_*.md | head -1)`

2. 仔细阅读完整偏差清单（不只是 Top 3），尤其留意：
   - 哪些维度（rating / style / theme_stage / conviction）出现重复警示
   - 哪些 horizon 命中率系统性偏低
   - 期望收益为负的评级（表示"赔钱"）

3. 给用户提出**具体的 prompt/阈值改动建议**，例如：
   - "theme_speculation+UNDERWEIGHT 在 T+1 命中率 28% → 建议把 Path B 阈值从 c≥50/m≥65 放宽到 c≥45/m≥60"
   - "SELL 在 T+5 期望 -4.2% → SELL 评级过度激进，建议在 Step 6 第三步加更严限制"

4. 询问用户是否要立即落地某条改动。

## 设计原则

- **cron 路径**：纯 Python 脚本完成 6 步，无 LLM 参与，可定时调度
- **skill 路径**：Python 脚本之上 + LLM 深度解读，给具体改动建议
- 两条路径共享 `weekly_review.py` 这个 single source of truth

## 异常处理

- `weekly_review.py` 退出码非 0 时，**飞书会收到失败消息**（脚本内部已处理）
- 如果连飞书都发不出来（webhook 故障），脚本会在终端打印错误；cron 调用时这会写到 cron log

## 调度配置

V1 用 macOS crontab：

```
0 10 * * 6 cd /Users/lailixiang/WorkSpace/QoderWorkspace/TradingAgents && .venv/bin/python -m tradingagents.harness.weekly_review >> harness_data/weekly_review.log 2>&1
```

加到 crontab：`crontab -e` 编辑后插入这行。

## 相关脚本

- `tradingagents/harness/weekly_review.py` — 主入口
- `.claude/skills/harness-weekly-review/send_feishu.sh` — 飞书 webhook helper
- `tradingagents/harness/backtest.py` — 切片统计
- `tradingagents/harness/review.py` — 偏差识别 + markdown 生成
